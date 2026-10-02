"""Experiment 3: how many of an LLM judge's "incorrect" verdicts are backed by a false claim in the answer?

Data: TruthfulQA (generation split, Apache-2.0), a fixed random sample of questions.
Systems under test answer in 2-3 sentences; two judges from different families grade every answer with the
evidence protocol in score_audit.evidence_judge. All LLM calls are cached in data/llm_cache.jsonl, so the
experiment re-runs offline. Outputs:
  results/exp3_items.csv          one row per (judge, system, question)
  results/exp3_summary.csv        raw vs evidence-backed error rate per judge x system, with Wilson CIs
  figures/fig3_evidence.{png,pdf} raw vs evidence-backed error rate
  figures/fig3b_unsupported.{png,pdf} why "incorrect" verdicts fail the evidence check
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from datasets import load_dataset

from score_audit.evidence_judge import Client, grade
from score_audit.plotting import BLUE, GRAY, RED, plt, results_dir, save
from score_audit.stats import cohens_kappa, wilson_ci

N_QUESTIONS = int(os.environ.get("N_QUESTIONS", 150))
SYSTEMS = ["meta-llama/llama-3.1-8b-instruct", "google/gemini-2.5-flash-lite", "mistralai/mistral-small-3.2-24b-instruct"]
JUDGES = ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"]
ANSWER_PROMPT = "Answer the question in 2-3 sentences.\nQuestion: {q}"
short = {s: s.split("/")[-1] for s in SYSTEMS + JUDGES}
ABBR = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "gemini-2.5-flash-lite": "Gemini-2.5-FL",
        "mistral-small-3.2-24b-instruct": "Mistral-Small-3.2", "gpt-4o-mini": "GPT-4o-mini"}


def main():
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    idx = np.random.default_rng(0).choice(len(ds), N_QUESTIONS, replace=False)
    qs = [ds[int(i)] for i in idx]
    client = Client()

    def answer(args):
        s, q = args
        return client.chat(s, ANSWER_PROMPT.format(q=q["question"]), max_tokens=200)

    jobs = [(s, q) for s in SYSTEMS for q in qs]
    with ThreadPoolExecutor(8) as ex:
        answers = list(ex.map(answer, jobs))

    def judge(args):
        j, (s, q), a = args
        ref = q["best_answer"] + " (also acceptable: " + "; ".join(q["correct_answers"]) + ")"
        g = grade(client, j, q["question"], ref, "; ".join(q["incorrect_answers"]), a)
        return {"judge": short[j], "system": short[s], "question": q["question"], "category": q["category"],
                "reference": ref, "false_answers": "; ".join(q["incorrect_answers"]), "answer": a, **g}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(judge, [(j, job, a) for j in JUDGES for job, a in zip(jobs, answers)]))
    df = pd.DataFrame(rows)
    out = results_dir()
    df.to_csv(out / "exp3_items.csv", index=False)

    summ = []
    for (j, s), d in df.groupby(["judge", "system"]):
        n = len(d)
        raw = int((d.verdict == "incorrect").sum())
        backed = int(d.evidence_backed_incorrect.sum())
        summ.append({"judge": j, "system": s, "n": n,
                     "raw_incorrect": raw, "raw_rate": raw / n, "raw_ci": wilson_ci(raw, n),
                     "backed_incorrect": backed, "backed_rate": backed / n, "backed_ci": wilson_ci(backed, n),
                     "share_unsupported": (raw - backed) / raw if raw else np.nan,
                     "no_quote": int(((d.verdict == "incorrect") & ~d.quote_found).sum()),
                     "quote_not_false": int(((d.verdict == "incorrect") & d.quote_found & (d.claim_contradicts == False)).sum())})  # noqa: E712
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "exp3_summary.csv", index=False)

    # agreement between the two judges, before and after the evidence check
    piv = {}
    for col in ("verdict", "audited_verdict"):
        p = df.pivot_table(index=["system", "question"], columns="judge", values=col, aggfunc="first")
        piv[col] = cohens_kappa(p[short[JUDGES[0]]], p[short[JUDGES[1]]])
    print("judge agreement (Cohen's kappa): raw =", round(piv["verdict"], 3), "| audited =", round(piv["audited_verdict"], 3))

    # fig 3: raw vs evidence-backed error rate with 95% CIs
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    labels = [f"{ABBR[r.system]}\njudge: {ABBR[r.judge]}" for r in sm.itertuples()]
    x = np.arange(len(sm))
    for off, col, rate, ci, lab in ((-0.18, RED, "raw_rate", "raw_ci", "judge says incorrect"),
                                    (0.18, BLUE, "backed_rate", "backed_ci", "incorrect AND false claim shown")):
        y = 100 * sm[rate].to_numpy()
        lo = y - 100 * np.array([c[0] for c in sm[ci]])
        hi = 100 * np.array([c[1] for c in sm[ci]]) - y
        ax.bar(x + off, y, width=0.36, color=col, label=lab)
        ax.errorbar(x + off, y, yerr=[lo, hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Error rate (%)")
    ax.set_title(f"TruthfulQA, {N_QUESTIONS} questions: \"incorrect\" rate before and after the evidence check")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig3_evidence")

    # fig 3b: why unsupported "incorrect" verdicts fail
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(labels, sm.no_quote, color=GRAY, label="quote not found in answer")
    ax.barh(labels, sm.quote_not_false, left=sm.no_quote, color=RED, label="quoted claim is not false")
    ax.set_xlabel('Unsupported "incorrect" verdicts (count)')
    ax.tick_params(axis="y", labelsize=7.5)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig3b_unsupported")
    print(sm.drop(columns=["raw_ci", "backed_ci"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
