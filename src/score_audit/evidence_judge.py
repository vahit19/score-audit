"""Evidence-backed grading: an LLM judge may only call an answer wrong if it can show the false claim.

Protocol for each (question, reference, answer):
  1. Holistic judge call returns verdict in {correct, incomplete, incorrect} and, if incorrect,
     a VERBATIM quote of the false claim.
  2. Code check: the quote must actually occur in the answer (whitespace/case-insensitive).
  3. Narrow claim check (second, short call): does this one claim contradict the reference?
An "incorrect" verdict that fails step 2 or 3 is recorded as unsupported. The audit reports how many
of a judge's "incorrect" verdicts survive, i.e. how much of a model's error rate is backed by evidence.

All calls go through a JSON cache on disk, so every published number can be re-derived offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from openai import OpenAI

CACHE = Path(__file__).resolve().parents[2] / "data" / "llm_cache.jsonl"


class Client:
    """OpenAI-compatible client with an append-only on-disk cache (offline replay when cached)."""

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1"):
        self._mem: dict[str, str] = {}
        self._lock = threading.Lock()
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self._mem[rec["key"]] = rec["content"]
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._api = OpenAI(api_key=key, base_url=base_url) if key else None

    def chat(self, model: str, prompt: str, json_mode: bool = False, max_tokens: int = 400) -> str:
        k = hashlib.sha256(json.dumps([model, prompt, json_mode, max_tokens]).encode()).hexdigest()
        if k in self._mem:
            return self._mem[k]
        if self._api is None:
            raise RuntimeError("not cached and no API key set")
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self._api.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                              messages=[{"role": "user", "content": prompt}], **kw)
        content = r.choices[0].message.content or ""
        with self._lock:
            self._mem[k] = content
            CACHE.parent.mkdir(exist_ok=True)
            with CACHE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": k, "model": model, "content": content}, ensure_ascii=False) + "\n")
        return content


JUDGE_PROMPT = """You are grading an answer against a reference.
Question: {question}
Reference (correct) answer: {reference}
Known false answers for this question: {false_answers}
Answer to grade: {answer}

Return JSON with keys:
  "verdict": "correct" if the answer's main claim agrees with the reference;
             "incomplete" if nothing it says is false but it misses or avoids the key point;
             "incorrect" if it states something false.
  "false_claim": if "incorrect", copy the false sentence VERBATIM from the answer; else "".
  "reasoning": one sentence."""

CLAIM_PROMPT = """Reference fact: {reference}
Claim: {claim}
Is the claim FALSE according to the reference? Leaving out a detail is not false.
Return JSON {{"contradicts": true}} or {{"contradicts": false}}."""


def _norm(s: str) -> str:
    return " ".join(s.split()).strip("\"'* ").lower()


def _parse(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        return json.loads(s[a:b + 1]) if a >= 0 < b else {}


def grade(client: Client, judge: str, question: str, reference: str, false_answers: str, answer: str) -> dict:
    v = _parse(client.chat(judge, JUDGE_PROMPT.format(question=question, reference=reference,
                                                      false_answers=false_answers, answer=answer), json_mode=True))
    verdict = v.get("verdict", "invalid")
    claim = v.get("false_claim") or ""
    quote_ok = bool(claim) and _norm(claim) in _norm(answer)
    contradicts = None
    if verdict == "incorrect" and quote_ok:
        c = _parse(client.chat(judge, CLAIM_PROMPT.format(reference=reference, claim=claim), json_mode=True, max_tokens=30))
        contradicts = bool(c.get("contradicts"))
    supported = verdict == "incorrect" and quote_ok and bool(contradicts)
    return {"verdict": verdict, "false_claim": claim, "quote_found": quote_ok, "claim_contradicts": contradicts,
            "evidence_backed_incorrect": supported,
            "audited_verdict": "incorrect" if supported else ("incomplete" if verdict == "incorrect" else verdict)}
