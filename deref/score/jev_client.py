"""Jev client (System One wire schema): POST {"state", "model", "questions"} to `{base}/v1/systemone`.

- Base URL: env JEV_BASE_URL (default https://openrouter.ai/api). API key: env JEV_KEY (sent in the Authorization header;
  never logged).
- The requested model is pinned (PINNED); the runner stops if any reply reports another model.
- Retries with exponential back-off on 429, 5xx (incl. Cloudflare 520-524) and transport errors; any other status
  raises clm.CLMError(status, body) (e.g. HTTP 400 "max_tokens_exceeded" for inputs over Jev's context).
- Optional sqlite answer cache, table ans(k TEXT PRIMARY KEY, v TEXT), keyed by cache_key("jev", model, state,
  questions) = sha1(json.dumps([name, model, state, questions], sort_keys=True, ensure_ascii=False)); value
  {"answers", "model"}. An interrupted run resumes from it; identical inputs are billed once.
- CostJudge also counts billed (non-cached) requests, enforces a cap on them and sums the reported usage.cost.
Score of one per-argument item: answers["q"]["noul"] for questions {"q": {"type": "noul", "instructions": question}},
with state = the item's context (no system prompt, no chat template; rendering is server-side).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

PINNED = "typesafe/jev-1.13-20260917"
SEED = 20260930
NAME = "jev"
DEFAULT_BASE_URL = "https://openrouter.ai/api"
RETRY_STATUS = {429, 500, 502, 503, 504, 529} | {520, 521, 522, 523, 524}


class Budget(Exception):
    """Raised when the cap on billed requests is reached."""


def cache_key(*parts: Any) -> str:
    """sha1 of the JSON list of parts (sort_keys=True, ensure_ascii=False)."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def questions_for(question):
    """The one-question payload used for every per-argument item."""
    return {"q": {"type": "noul", "instructions": question}}


def item_key(row, model=PINNED):
    """Cache key of a judge-input row {"context", "question"}."""
    return cache_key(NAME, model, row["context"], questions_for(row["question"]))


def request_body(state, questions, model=PINNED):
    """The JSON body POSTed to {base}/v1/systemone."""
    return {"state": state, "model": model, "questions": questions}


def percentile(values, q):
    """``q``-th percentile of a non-empty sequence (nearest rank)."""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


@dataclass
class Reply:
    answers: dict
    latency_ms: float          # client-side wall-clock of the successful request (0 when cached)
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False
    retries: int = 0


@dataclass
class Stats:
    requests: int = 0
    cached: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list = field(default_factory=list)

    def summary(self) -> dict:
        lat = self.latencies_ms
        return {"requests": self.requests, "cached": self.cached, "retries": self.retries,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
                "latency_ms_p50": round(percentile(lat, 0.5), 1) if lat else None,
                "latency_ms_p95": round(percentile(lat, 0.95), 1) if lat else None}


class Judge:
    """``ask(state, questions)`` against the Jev endpoint, with retries, timing and a cache."""

    def __init__(self, cache: str | None = None, timeout: float = 120.0, max_retries: int = 6,
                 model: str = PINNED, base_url: str | None = None, key: str | None = None):
        from clm import CLMClient
        self.name = NAME
        key = key or os.environ.get("JEV_KEY")
        if not key:
            raise RuntimeError("set the environment variable JEV_KEY")
        self.model = model
        self.base_url = base_url or os.environ.get("JEV_BASE_URL", DEFAULT_BASE_URL)
        self.client = CLMClient(self.base_url, key, timeout, self.model)  # (base_url, key, timeout, model)
        self.max_retries = max_retries
        self.stats = Stats()
        self._lock = threading.Lock()
        self.db = None
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            self.db = sqlite3.connect(cache, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS ans (k TEXT PRIMARY KEY, v TEXT)")

    def _post(self, state, questions) -> dict:
        """Raw response body, retried on rate limits, 5xx and transport errors."""
        from clm import CLMError
        body = request_body(state, questions, self.model)
        url = f"{self.client.base_url}/v1/systemone"
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client._s.post(url, json=body, timeout=self.client.timeout)
                if r.status_code == 200:
                    return r.json() | {"_retries": attempt}
                if r.status_code not in RETRY_STATUS:
                    raise CLMError(r.status_code, r.text[:300])
                last = CLMError(r.status_code, r.text[:300])
                wait = float(r.headers.get("Retry-After") or 0) or min(30.0, 0.5 * 2 ** attempt)
            except requests.RequestException as e:
                last = e
                wait = min(30.0, 0.5 * 2 ** attempt)
            time.sleep(wait + random.uniform(0, 0.25))
        raise RuntimeError(f"{self.name}: request failed after {self.max_retries} retries: {last}")

    def ask(self, state, questions) -> Reply:
        k = cache_key(self.name, self.model, state, questions)
        if self.db is not None:
            with self._lock:
                row = self.db.execute("SELECT v FROM ans WHERE k=?", (k,)).fetchone()
            if row:
                j = json.loads(row[0])
                with self._lock:
                    self.stats.cached += 1
                return Reply(j["answers"], 0.0, 0, 0, j.get("model", self.model), cached=True)
        t0 = time.perf_counter()
        j = self._post(state, questions)
        ms = (time.perf_counter() - t0) * 1000
        usage = j.get("usage") or {}
        reply = Reply(j["answers"], ms, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
                      j.get("model", self.model), retries=int(j.get("_retries", 0)))
        with self._lock:
            s = self.stats
            s.requests += 1
            s.retries += reply.retries
            s.input_tokens += reply.input_tokens
            s.output_tokens += reply.output_tokens
            s.latencies_ms.append(ms)
            if self.db is not None:
                self.db.execute("INSERT OR REPLACE INTO ans VALUES (?,?)",
                                (k, json.dumps({"answers": reply.answers, "model": reply.model}, ensure_ascii=False)))
                self.db.commit()
        return reply

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


class CostJudge(Judge):
    """Judge that also tallies the reported cost and enforces a cap on billed requests (0 = no cap)."""

    def __init__(self, *a, max_calls=0, **k):
        super().__init__(*a, **k)
        self.cost, self.billed, self.cap = 0.0, 0, max_calls

    def _post(self, state, questions):
        with self._lock:
            if self.cap and self.billed >= self.cap:
                raise Budget(f"max billed calls reached ({self.cap})")
            self.billed += 1
        j = super()._post(state, questions)
        with self._lock:
            self.cost += float((j.get("usage") or {}).get("cost") or 0.0)
        return j


def make_judge(cache, max_calls, model=PINNED):
    """CostJudge on the pinned model with the given sqlite cache."""
    return CostJudge(cache=str(cache), model=model, max_calls=max_calls)


def snapshot(cache):
    """Read-only in-memory copy of the sqlite cache."""
    src = sqlite3.connect(f"file:{cache}?mode=ro", uri=True, timeout=600)
    db = sqlite3.connect(":memory:")
    src.backup(db)
    src.close()
    return db
