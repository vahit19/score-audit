"""Experiment 4: compare raw and evidence-backed judge verdicts against HUMAN truthfulness labels.

Human labels: TruthfulQA's released human evaluations (data/finetune_truth.jsonl in the TruthfulQA repository,
Apache-2.0): model answers labelled true / not true by people. Rows whose answer is one of the dataset's own
reference answers are dropped, so only human-graded MODEL answers remain. A fixed random sample is stored in
data/truthfulqa_human_sample.jsonl for offline re-runs.

For each judge we report, against the human label "not true":
  precision and recall of the raw "incorrect" verdict, and of the evidence-backed "incorrect" verdict.
Outputs: results/exp4_items.csv, results/exp4_metrics.csv, figures/fig4_human_labels.{png,pdf}
"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from score_audit.evidence_judge import Client, grade
from score_audit.plotting import BLUE, RED, plt, results_dir, save
from score_audit.stats import wilson_ci

N = int(os.environ.get("N_HUMAN", 300))
JUDGES = ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"]
SRC = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/finetune_truth.jsonl"
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "truthfulqa_human_sample.jsonl"
ABBR = {"gpt-4o-mini": "GPT-4o-mini", "gemini-2.5-flash-lite": "Gemini-2.5-FL"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def build_sample() -> list[dict]:
    if SAMPLE.exists():
        return [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    meta = {r["question"]: r for r in ds}
    rows = []
    for line in urllib.request.urlopen(SRC).read().decode("utf-8").splitlines():
        r = json.loads(line)
        q, a = r["prompt"][3:].split("\nA: ", 1)
        a = a.rsplit("\nTrue:", 1)[0].strip()
        m = meta.get(q)
        if m is None or _norm(a) in {_norm(x) for x in m["correct_answers"] + m["incorrect_answers"] + [m["best_answer"]]}:
            continue                                   # keep only human-graded model answers
        rows.append({"question": q, "answer": a, "human_true": r["completion"].strip() == "yes",
                     "reference": m["best_answer"] + " (also acceptable: " + "; ".join(m["correct_answers"]) + ")",
                     "false_answers": "; ".join(m["incorrect_answers"])})
    rng = np.random.default_rng(0)
    pick = [rows[i] for i in rng.choice(len(rows), N, replace=False)]
    SAMPLE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pick) + "\n", encoding="utf-8")
    print(f"human-graded model answers available: {len(rows)}; sampled {N}")
    return pick


def pr(pred: pd.Series, truth: pd.Series) -> dict:
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    return {"precision": tp / (tp + fp) if tp + fp else np.nan, "prec_ci": wilson_ci(tp, tp + fp),
            "recall": tp / (tp + fn) if tp + fn else np.nan, "rec_ci": wilson_ci(tp, tp + fn), "flagged": tp + fp}


def main():
    items = build_sample()
    client = Client()

    def job(args):
        j, r = args
        g = grade(client, j, r["question"], r["reference"], r["false_answers"], r["answer"])
        return {"judge": j.split("/")[-1], **r, **g}

    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(job, [(j, r) for j in JUDGES for r in items])))
    out = results_dir()
    df.to_csv(out / "exp4_items.csv", index=False)

    truth_false = ~df.human_true
    rows = []
    for j, d in df.groupby("judge"):
        t = ~d.human_true
        for name, pred in (("raw judge", d.verdict == "incorrect"), ("evidence-backed", d.evidence_backed_incorrect.astype(bool))):
            rows.append({"judge": j, "verdict": name, "n": len(d), "human_not_true": int(t.sum()), **pr(pred, t)})
    met = pd.DataFrame(rows)
    met.to_csv(out / "exp4_metrics.csv", index=False)
    print(f"human 'not true' rate in sample: {truth_false.mean():.3f}")
    print(met.drop(columns=["prec_ci", "rec_ci"]).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, metric, ci in ((axes[0], "precision", "prec_ci"), (axes[1], "recall", "rec_ci")):
        x = np.arange(len(JUDGES))
        for off, (name, col) in zip((-0.18, 0.18), (("raw judge", RED), ("evidence-backed", BLUE))):
            d = met[met.verdict == name].set_index("judge").loc[[j.split("/")[-1] for j in JUDGES]]
            y = 100 * d[metric].to_numpy()
            lo = y - 100 * np.array([c[0] for c in d[ci]])
            hi = 100 * np.array([c[1] for c in d[ci]]) - y
            ax.bar(x + off, y, 0.36, color=col, label=name)
            ax.errorbar(x + off, y, yerr=[lo, hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
        ax.set_xticks(x, [ABBR[j.split("/")[-1]] for j in JUDGES])
        ax.set_title(f"{metric.capitalize()} vs human labels")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel('"Incorrect" verdicts (%)')
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, "fig4_human_labels")


if __name__ == "__main__":
    main()
