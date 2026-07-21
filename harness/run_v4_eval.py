#!/usr/bin/env python3
# REFERENCE RUNNER (not standalone). Assumes the original repo layout (eval/, src/)
# and the pipeline source + indices, released at camera-ready. This bundle is a
# result-verification package; use harness/verify_claims.py to recompute headline stats.
"""
NLU v4 Eval Runner — Core 17 Intent Evaluation
=================================================
Runs the v4_core17_dataset through PipelineV4 and produces:
  - Per-intent accuracy (precision / recall / F1)
  - Tier breakdown (clean vs messy vs adversarial)
  - Confusion matrix (top confusions)
  - Path breakdown (hot vs warm vs cold)
  - OOS detection accuracy
  - Multi-intent detection rate
  - Latency stats
  - Saves results as JSON for tracking over time

Usage:
    python eval/run_v4_eval.py                # Full run
    python eval/run_v4_eval.py --save         # Save results to eval/results/
    python eval/run_v4_eval.py --verbose      # Show every query result
    python eval/run_v4_eval.py --tier messy   # Only run messy tier
    python eval/run_v4_eval.py --intent cc_block_card  # Only one intent
    python eval/run_v4_eval.py --json         # JSON output only
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path nlup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.v4_core17_dataset import (
    EVAL_QUERIES, OOS_QUERIES, MULTI_INTENT_QUERIES, CORE_17, EvalQuery,
    ALL_INTENTS, BOUNDARY_QUERIES,
)
from src.pipeline_v4 import NLUPipelineV4
from src.models.schemas import ActionType, ClassificationSource


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    query: str
    expected_intent: str
    actual_intent: str
    actual_domain: str
    confidence: float
    source: str
    action: str
    correct: bool
    tier: str
    latency_ms: float
    slm_invoked: bool = False
    slm_intent: Optional[str] = None
    notes: str = ""


@dataclass
class IntentMetrics:
    intent: str
    total: int = 0
    correct: int = 0
    # Per tier
    clean_total: int = 0
    clean_correct: int = 0
    messy_total: int = 0
    messy_correct: int = 0
    adversarial_total: int = 0
    adversarial_correct: int = 0
    # Confused with
    confused_with: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def clean_accuracy(self) -> float:
        return self.clean_correct / self.clean_total if self.clean_total else 0.0

    @property
    def messy_accuracy(self) -> float:
        return self.messy_correct / self.messy_total if self.messy_total else 0.0

    @property
    def adversarial_accuracy(self) -> float:
        return self.adversarial_correct / self.adversarial_total if self.adversarial_total else 0.0


# ---------------------------------------------------------------------------
# Eval Runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ablation stubs (W-NUT paper experiment) — injected into the pipeline's
# lazy-init slots AFTER construction. src/ is never modified; removing the
# --ablate flag restores the exact production pipeline.
# ---------------------------------------------------------------------------
class _IdentityNormalizer:
    """Layer 2 off: pass raw text through untouched."""
    def normalize(self, text: str, language_override=None):
        from src.models.schemas import NormalizedQuery
        return NormalizedQuery(original=text, normalized=text)


class _NullRulesEngine:
    """Layer 6 off: no deterministic rule matches ever fire."""
    def match(self, text: str):
        return None

    def match_extended(self, text: str):
        return None

    def match_all(self, text: str):
        return []


class _NullSLMReranker:
    """Layer 11 off: reranker never runs; decision falls back to retrieval top-1."""
    def rerank(self, *args, **kwargs):
        return None


ABLATABLE = {
    "normalizer": ("_normalizer", _IdentityNormalizer),
    "rules": ("_rules_engine", _NullRulesEngine),
    "slm": ("_slm_reranker", _NullSLMReranker),
}


class _RerankingRetriever:
    """TRUE reranker mode (user-correct design): wraps the hybrid retriever;
    rescores top-K candidates with the cross-encoder, blends with retrieval
    scores, reorders — downstream layers (cliff/OOS/decision/SLM) unchanged.
    """

    def __init__(self, inner, model_name="BAAI/bge-reranker-v2-m3",
                 descs=None, top_k=8, alpha=0.5):
        from sentence_transformers import CrossEncoder
        self.inner = inner
        self.model = CrossEncoder(model_name, max_length=256)
        self.descs = descs or {}
        self.top_k = top_k
        self.alpha = alpha  # weight of xenc score in the blend
        self.stats = {"rerank_calls": 0, "top1_changed": 0}
        self.qlog = []  # per-query calibration log

    def retrieve(self, *args, **kwargs):
        import numpy as _np
        result = self.inner.retrieve(*args, **kwargs)
        cands = getattr(result, "candidates", None)
        if not cands or len(cands) < 2:
            return result
        self.stats["rerank_calls"] += 1
        head = cands[: self.top_k]
        pairs = []
        for c in head:
            text = self.descs.get(c.intent, c.intent.replace("_", " "))
            if c.matched_example:
                text = f"{text}. e.g. {c.matched_example}"
            pairs.append((kwargs.get("query") or args[0], text))
        xs = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        xs = _np.asarray(xs, dtype=float)
        # min-max normalize both score sets within the head window
        rs = _np.asarray([c.score for c in head], dtype=float)
        def _norm(v):
            span = v.max() - v.min()
            return (v - v.min()) / span if span > 1e-9 else _np.zeros_like(v)
        blended = (1 - self.alpha) * _norm(rs) + self.alpha * _norm(xs)
        old_top = head[0].intent
        order = _np.argsort(blended)[::-1]
        new_head = [head[int(i)] for i in order]
        # write blended scores back (rescaled into the head's original range
        # so downstream cliff/OOS thresholds stay meaningful)
        lo, hi = float(rs.min()), float(rs.max())
        for c, b in zip(new_head, sorted(blended, reverse=True)):
            c.score = lo + b * (hi - lo) if hi > lo else c.score
        result.candidates = new_head + cands[self.top_k:]
        if new_head[0].intent != old_top:
            self.stats["top1_changed"] += 1
        # calibration / recall log
        try:
            import numpy as _np2
            _q = kwargs.get("query") or (args[0] if args else "")
            self.qlog.append({
                "query": str(_q)[:120],
                "pre_top5": [(c.intent, round(float(s0), 4)) for c, s0 in zip(head, rs.tolist())][:5],
                "post_top5": [(c.intent, round(float(c.score), 4)) for c in new_head[:5]],
                "xenc_raw_top5": [round(float(x), 4) for x in sorted(xs.tolist(), reverse=True)[:5]],
                "xenc_sigmoid_top": round(float(1.0 / (1.0 + _np2.exp(-max(xs)))), 4),
                "margin_blend": round(float(sorted(blended, reverse=True)[0] - sorted(blended, reverse=True)[1]), 4) if len(blended) > 1 else 1.0,
                "top1_changed": new_head[0].intent != old_top,
            })
        except Exception:
            pass
        # keep result.top_intent/top_score consistent if present
        if hasattr(result, "top_intent"):
            try:
                result.top_intent = new_head[0].intent
                result.top_score = new_head[0].score
            except Exception:
                pass
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _XencReranker:
    """Cross-encoder reranker injected into the SLM slot (W-NUT Track 2).

    mode="replace": cross-encoder answers every escalated query (no LLM).
    mode="cascade": cross-encoder answers when its softmax margin >= threshold;
                    low-margin queries fall through to the real SLM.
    Scoring text per candidate: "<intent description>. e.g. <matched_example>".
    """

    def __init__(self, mode: str = "replace", margin: float = 0.15,
                 model_name: str = "BAAI/bge-reranker-v2-m3",
                 real_slm=None):
        from sentence_transformers import CrossEncoder
        self.mode = mode
        self.margin = margin
        self.real_slm = real_slm
        self.model = CrossEncoder(model_name, max_length=256)
        self.stats = {"calls": 0, "cascaded_to_slm": 0}

    def rerank(self, query, candidates, intent_descriptions=None,
               force=False, max_candidates=None, user_context_hint=None):
        import numpy as _np
        from src.models.schemas import SLMResult
        t0 = time.perf_counter()
        self.stats["calls"] += 1
        cands = candidates[: (max_candidates or 8)]
        if not cands:
            return SLMResult(intent="__uncertain__", confidence=0.0,
                             was_invoked=True, reasoning="no candidates")
        descs = intent_descriptions or {}
        pairs = []
        for c in cands:
            text = descs.get(c.intent, c.intent.replace("_", " "))
            if c.matched_example:
                text = f"{text}. e.g. {c.matched_example}"
            pairs.append((query, text))
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        order = _np.argsort(scores)[::-1]
        exp = _np.exp(scores - _np.max(scores))
        probs = exp / exp.sum()
        top, second = order[0], (order[1] if len(order) > 1 else order[0])
        top_margin = float(probs[top] - probs[second]) if len(order) > 1 else 1.0
        # Confidence = sigmoid of the top raw logit (bge-reranker relevance),
        # NOT the softmax prob — softmax over 8 candidates rarely clears the
        # pipeline's 0.5/0.85 confidence gates and the result gets discarded.
        top_conf = float(1.0 / (1.0 + _np.exp(-scores[top])))
        latency = (time.perf_counter() - t0) * 1000

        if self.mode == "cascade" and top_margin < self.margin and self.real_slm:
            self.stats["cascaded_to_slm"] += 1
            return self.real_slm.rerank(
                query, candidates, intent_descriptions=intent_descriptions,
                force=force, max_candidates=max_candidates,
                user_context_hint=user_context_hint,
            )
        return SLMResult(
            intent=cands[int(top)].intent,
            confidence=top_conf,
            reasoning=f"xenc margin={top_margin:.3f} sigmoid={top_conf:.3f}",
            was_invoked=True,
            model_name="bge-reranker-v2-m3",
            latency_ms=latency,
        )


class V4EvalRunner:
    def __init__(self, verbose: bool = False, ablate: Optional[List[str]] = None,
                 reranker_mode: Optional[str] = None, reranker_margin: float = 0.15,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.verbose = verbose
        self.ablate = ablate or []
        self.reranker_mode = reranker_mode
        self.reranker_margin = reranker_margin
        self.reranker_model = reranker_model
        self.xenc = None
        self.pipeline: Optional[NLUPipelineV4] = None
        self.results: List[QueryResult] = []
        self.oos_results: List[QueryResult] = []
        self.multi_results: List[Dict] = []
        self.boundary_results: List[QueryResult] = []
        self.intent_metrics: Dict[str, IntentMetrics] = {}
        self.start_time = None

    def nlup(self):
        """Initialize pipeline."""
        print("=" * 70)
        print("  NLU v4 EVAL — Core 17 Intent Evaluation")
        print("=" * 70)
        print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Queries:  {len(EVAL_QUERIES)} core + {len(OOS_QUERIES)} OOS + {len(MULTI_INTENT_QUERIES)} multi + {len(BOUNDARY_QUERIES)} boundary")
        print()

        print("  Loading pipeline...", end=" ", flush=True)
        t0 = time.perf_counter()
        self.pipeline = NLUPipelineV4()
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"done ({load_ms:.0f}ms)")

        # Apply ablations by pre-filling the lazy-init slots with stubs
        for name in self.ablate:
            attr, stub_cls = ABLATABLE[name]
            setattr(self.pipeline, attr, stub_cls())
            print(f"  ABLATED: {name} (layer stubbed out)")

        # TRUE rerank mode: wrap the retriever, leave everything else alone
        if self.reranker_mode == "rerank":
            inner = self.pipeline._get_hybrid_retriever()
            print(f"  RERANKER (list-rerank): {self.reranker_model} "
                  f"alpha={self.reranker_margin} (loading...)", flush=True)
            self.xenc = _RerankingRetriever(
                inner, model_name=self.reranker_model,
                descs=self.pipeline._taxonomy.intent_descriptions,
                alpha=self.reranker_margin,
            )
            self.pipeline._hybrid_retriever = self.xenc
            print("  RERANKER loaded")
        # Referee modes: cross-encoder injected into the SLM slot
        elif self.reranker_mode:
            real_slm = None
            if self.reranker_mode == "cascade":
                real_slm = self.pipeline._get_slm_reranker()
            print(f"  RERANKER: bge-reranker-v2-m3 mode={self.reranker_mode} "
                  f"margin={self.reranker_margin} (loading...)", flush=True)
            self.xenc = _XencReranker(
                mode=self.reranker_mode, margin=self.reranker_margin,
                model_name=self.reranker_model, real_slm=real_slm,
            )
            self.pipeline._slm_reranker = self.xenc
            print("  RERANKER loaded")

        print("  Warming up...", end=" ", flush=True)
        t0 = time.perf_counter()
        self.pipeline.classify("warmup query")
        warmup_ms = (time.perf_counter() - t0) * 1000
        print(f"done ({warmup_ms:.0f}ms)")
        print()

    # ----- Core intent evaluation -----

    def run_core(self, queries: List[EvalQuery]):
        """Run core intent queries."""
        print("-" * 70)
        print("  CORE 17 INTENT EVALUATION")
        print("-" * 70)

        for q in queries:
            result = self._classify_query(q)
            self.results.append(result)
            self._update_metrics(result)

            if self.verbose or not result.correct:
                tag = "Y" if result.correct else "N"
                slm = f" SLM:{result.slm_intent}" if result.slm_invoked else ""
                print(
                    f"  {tag}  [{result.tier:5s}] {q.text:50s} "
                    f"exp={q.expected_intent:25s} got={result.actual_intent:25s} "
                    f"{result.confidence:.2f} {result.source:10s} {result.latency_ms:5.0f}ms{slm}"
                )

    def _classify_query(self, q: EvalQuery) -> QueryResult:
        """Classify a single query and return result."""
        t0 = time.perf_counter()
        r = self.pipeline.classify(q.text)
        ms = (time.perf_counter() - t0) * 1000

        slm_invoked = False
        slm_intent = None
        if r.trace and r.trace.slm_result and r.trace.slm_result.was_invoked:
            slm_invoked = True
            slm_intent = r.trace.slm_result.intent

        return QueryResult(
            query=q.text,
            expected_intent=q.expected_intent,
            actual_intent=r.intent,
            actual_domain=r.domain,
            confidence=r.confidence,
            source=r.source.value,
            action=r.action.value,
            correct=(r.intent == q.expected_intent),
            tier=q.tier,
            latency_ms=ms,
            slm_invoked=slm_invoked,
            slm_intent=slm_intent,
            notes=q.notes,
        )

    def _update_metrics(self, r: QueryResult):
        """Update per-intent metrics."""
        intent = r.expected_intent
        if intent not in self.intent_metrics:
            self.intent_metrics[intent] = IntentMetrics(intent=intent)

        m = self.intent_metrics[intent]
        m.total += 1
        if r.correct:
            m.correct += 1

        # Per tier
        if r.tier == "clean":
            m.clean_total += 1
            m.clean_correct += int(r.correct)
        elif r.tier == "messy":
            m.messy_total += 1
            m.messy_correct += int(r.correct)
        elif r.tier == "adversarial":
            m.adversarial_total += 1
            m.adversarial_correct += int(r.correct)

        # Confusion tracking
        if not r.correct:
            m.confused_with[r.actual_intent] += 1

        # Running averages
        n = m.total
        m.avg_latency_ms = ((m.avg_latency_ms * (n - 1)) + r.latency_ms) / n
        m.avg_confidence = ((m.avg_confidence * (n - 1)) + r.confidence) / n

    # ----- OOS evaluation -----

    def run_oos(self, queries: List[EvalQuery]):
        """Run OOS queries — these should NOT match any core 17 intent."""
        print()
        print("-" * 70)
        print("  OOS DETECTION")
        print("-" * 70)

        for q in queries:
            t0 = time.perf_counter()
            r = self.pipeline.classify(q.text)
            ms = (time.perf_counter() - t0) * 1000

            # OOS is correct if:
            # 1. action is OOS/fallback, OR
            # 2. intent is oos_general/discovery, OR
            # 3. intent is NOT one of the core 17 (since those are the only ones that matter)
            is_oos_action = r.action in (ActionType.OOS, ActionType.CLARIFY)
            is_oos_intent = r.intent not in ALL_INTENTS
            correct = is_oos_action or is_oos_intent

            result = QueryResult(
                query=q.text,
                expected_intent="oos_general",
                actual_intent=r.intent,
                actual_domain=r.domain,
                confidence=r.confidence,
                source=r.source.value,
                action=r.action.value,
                correct=correct,
                tier="oos",
                latency_ms=ms,
                notes=q.notes,
            )
            self.oos_results.append(result)

            if self.verbose or not correct:
                tag = "Y" if correct else "N"
                print(
                    f"  {tag}  {q.text:50s} → {r.intent:25s} "
                    f"{r.action.value:10s} {r.confidence:.2f} {ms:5.0f}ms  ({q.notes})"
                )

    # ----- Boundary evaluation (explore vs action) -----

    def run_boundary(self, queries: List[EvalQuery]):
        """Run boundary queries — confusion group stress tests."""
        print()
        print("-" * 70)
        print("  BOUNDARY: EXPLORE vs ACTION DISAMBIGUATION")
        print("-" * 70)

        for q in queries:
            result = self._classify_query(q)
            self.boundary_results.append(result)

            tag = "Y" if result.correct else "N"
            slm = f" SLM:{result.slm_intent}" if result.slm_invoked else ""
            print(
                f"  {tag}  {q.text:50s} "
                f"exp={q.expected_intent:25s} got={result.actual_intent:25s} "
                f"{result.confidence:.2f}{slm}"
            )

    # ----- Multi-intent evaluation -----

    def run_multi(self, queries: List):
        """Run multi-intent queries."""
        print()
        print("-" * 70)
        print("  MULTI-INTENT DETECTION")
        print("-" * 70)

        for mq in queries:
            t0 = time.perf_counter()
            r = self.pipeline.classify(mq.text)
            ms = (time.perf_counter() - t0) * 1000

            has_secondary = r.secondary_hint is not None
            primary_match = r.intent == mq.expected_primary
            secondary_match = (r.secondary_hint == mq.expected_secondary) if has_secondary else False

            # Check multi-intent detection from trace
            is_multi = False
            if r.trace and r.trace.multi_intent_result:
                is_multi = r.trace.multi_intent_result.is_multi

            result = {
                "query": mq.text,
                "expected_primary": mq.expected_primary,
                "expected_secondary": mq.expected_secondary,
                "actual_primary": r.intent,
                "actual_secondary": r.secondary_hint,
                "multi_detected": is_multi,
                "primary_match": primary_match,
                "secondary_match": secondary_match,
                "latency_ms": ms,
                "notes": mq.notes,
            }
            self.multi_results.append(result)

            tag = "Y" if (primary_match and is_multi) else "~" if primary_match else "N"
            sec_str = r.secondary_hint or "none"
            print(
                f"  {tag}  {mq.text:55s} "
                f"P:{r.intent:25s} S:{sec_str:25s} multi={is_multi} {ms:.0f}ms"
            )

    # ----- Reporting -----

    def print_summary(self):
        """Print full summary report."""
        print()
        print("=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)

        # --- Overall accuracy ---
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        print(f"\n  OVERALL: {correct}/{total} = {correct/total*100:.1f}%")

        # --- Per-tier accuracy ---
        print(f"\n  PER-TIER ACCURACY:")
        for tier in ["clean", "messy", "adversarial"]:
            tier_results = [r for r in self.results if r.tier == tier]
            tier_correct = sum(1 for r in tier_results if r.correct)
            tier_total = len(tier_results)
            pct = tier_correct / tier_total * 100 if tier_total else 0
            bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
            print(f"    {tier:12s}  {tier_correct:3d}/{tier_total:3d} = {pct:5.1f}%  |{bar}|")

        # --- Per-intent accuracy ---
        print(f"\n  PER-INTENT ACCURACY:")
        sorted_intents = sorted(
            self.intent_metrics.values(),
            key=lambda m: m.accuracy,
            reverse=True
        )
        for m in sorted_intents:
            icon = "+" if m.accuracy >= 0.80 else "~" if m.accuracy >= 0.60 else "X"
            print(
                f"    {icon} {m.intent:25s}  {m.correct:2d}/{m.total:2d} = {m.accuracy*100:5.1f}%  "
                f"(C:{m.clean_accuracy*100:.0f}% M:{m.messy_accuracy*100:.0f}% A:{m.adversarial_accuracy*100:.0f}%)  "
                f"avg {m.avg_latency_ms:.0f}ms  conf {m.avg_confidence:.2f}"
            )

        # --- Confusion matrix (top confusions) ---
        print(f"\n  TOP CONFUSIONS:")
        all_confusions = []
        for m in self.intent_metrics.values():
            for confused_intent, count in m.confused_with.items():
                all_confusions.append((m.intent, confused_intent, count))
        all_confusions.sort(key=lambda x: x[2], reverse=True)

        for expected, got, count in all_confusions[:15]:
            print(f"    {expected:25s} → {got:25s}  ({count}x)")

        if not all_confusions:
            print("    (none — perfect accuracy!)")

        # --- Path breakdown ---
        print(f"\n  PATH BREAKDOWN:")
        sources = defaultdict(list)
        for r in self.results:
            sources[r.source].append(r)

        for source, results in sorted(sources.items()):
            src_correct = sum(1 for r in results if r.correct)
            src_total = len(results)
            avg_lat = sum(r.latency_ms for r in results) / src_total if src_total else 0
            slm_count = sum(1 for r in results if r.slm_invoked)
            print(
                f"    {source:12s}  {src_total:3d} queries  "
                f"{src_correct}/{src_total} correct ({src_correct/src_total*100:.0f}%)  "
                f"avg {avg_lat:.0f}ms  SLM:{slm_count}"
            )

        # --- SLM analysis ---
        slm_results = [r for r in self.results if r.slm_invoked]
        if slm_results:
            slm_correct = sum(1 for r in slm_results if r.correct)
            print(f"\n  SLM RERANKER:")
            print(f"    Invoked: {len(slm_results)}/{total} ({len(slm_results)/total*100:.1f}%)")
            print(f"    Accuracy when invoked: {slm_correct}/{len(slm_results)} ({slm_correct/len(slm_results)*100:.0f}%)")

        # --- OOS results ---
        if self.oos_results:
            oos_correct = sum(1 for r in self.oos_results if r.correct)
            oos_total = len(self.oos_results)
            print(f"\n  OOS DETECTION:")
            print(f"    {oos_correct}/{oos_total} = {oos_correct/oos_total*100:.1f}% correctly rejected")
            false_positives = [r for r in self.oos_results if not r.correct]
            if false_positives:
                print(f"    False positives (OOS misclassified as core intent):")
                for r in false_positives:
                    print(f"      \"{r.query}\" → {r.actual_intent} ({r.confidence:.2f})")

        # --- Multi-intent results ---
        if self.multi_results:
            multi_detected = sum(1 for r in self.multi_results if r["multi_detected"])
            primary_correct = sum(1 for r in self.multi_results if r["primary_match"])
            both_correct = sum(1 for r in self.multi_results if r["primary_match"] and r["secondary_match"])
            total_multi = len(self.multi_results)
            print(f"\n  MULTI-INTENT:")
            print(f"    Detected as multi:   {multi_detected}/{total_multi} ({multi_detected/total_multi*100:.0f}%)")
            print(f"    Primary correct:     {primary_correct}/{total_multi} ({primary_correct/total_multi*100:.0f}%)")
            print(f"    Both correct:        {both_correct}/{total_multi} ({both_correct/total_multi*100:.0f}%)")

        # --- Boundary results ---
        if self.boundary_results:
            bnd_correct = sum(1 for r in self.boundary_results if r.correct)
            bnd_total = len(self.boundary_results)
            print(f"\n  BOUNDARY (Explore vs Action):")
            print(f"    {bnd_correct}/{bnd_total} = {bnd_correct/bnd_total*100:.1f}% correctly disambiguated")
            bnd_failures = [r for r in self.boundary_results if not r.correct]
            if bnd_failures:
                print(f"    Failures:")
                for r in bnd_failures:
                    print(f"      \"{r.query}\" → exp={r.expected_intent} got={r.actual_intent} ({r.confidence:.2f})")

        # --- Latency ---
        latencies = [r.latency_ms for r in self.results]
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
        avg = sum(latencies) / len(latencies) if latencies else 0
        print(f"\n  LATENCY:")
        print(f"    Avg: {avg:.0f}ms  P50: {p50:.0f}ms  P95: {p95:.0f}ms  P99: {p99:.0f}ms")

        # --- Failures list ---
        failures = [r for r in self.results if not r.correct]
        if failures:
            print(f"\n  FAILURES ({len(failures)}):")
            for r in failures:
                notes = f"  ({r.notes})" if r.notes else ""
                print(
                    f"    [{r.tier:5s}] \"{r.query}\""
                    f"\n           expected={r.expected_intent}  got={r.actual_intent}  "
                    f"conf={r.confidence:.2f}  src={r.source}{notes}"
                )

        print()
        print("=" * 70)

    def save_results(self, output_dir: str = "eval/results"):
        """Save results as JSON."""
        out_path = Path(PROJECT_ROOT) / output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        _abl_tag = ("_ablate-" + "-".join(self.ablate)) if self.ablate else ""
        if self.reranker_mode:
            _abl_tag += f"_xenc-{self.reranker_mode}{self.reranker_margin}"
        filename = out_path / f"eval_{timestamp}{_abl_tag}.json"

        # Build summary
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)

        intent_summary = {}
        for m in self.intent_metrics.values():
            intent_summary[m.intent] = {
                "accuracy": round(m.accuracy, 4),
                "total": m.total,
                "correct": m.correct,
                "clean_accuracy": round(m.clean_accuracy, 4),
                "messy_accuracy": round(m.messy_accuracy, 4),
                "adversarial_accuracy": round(m.adversarial_accuracy, 4),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "avg_confidence": round(m.avg_confidence, 4),
                "confusions": dict(m.confused_with),
            }

        data = {
            "timestamp": timestamp,
            "ablated": self.ablate,
            "rerank_log": (getattr(self.xenc, "qlog", None) if self.reranker_mode == "rerank" else None),
            "reranker": (
                {"mode": self.reranker_mode, "margin": self.reranker_margin,
                 "model": self.reranker_model,
                 **(self.xenc.stats if self.xenc else {})}
                if self.reranker_mode else None
            ),
            "overall_accuracy": round(correct / total, 4) if total else 0,
            "total_queries": total,
            "correct": correct,
            "per_intent": intent_summary,
            "tier_accuracy": {
                tier: round(
                    sum(1 for r in self.results if r.tier == tier and r.correct)
                    / max(1, sum(1 for r in self.results if r.tier == tier)),
                    4,
                )
                for tier in ["clean", "messy", "adversarial"]
            },
            "oos_accuracy": round(
                sum(1 for r in self.oos_results if r.correct) / max(1, len(self.oos_results)),
                4,
            ) if self.oos_results else None,
            "latency": {
                "avg_ms": round(sum(r.latency_ms for r in self.results) / max(1, total), 1),
                "p50_ms": round(sorted(r.latency_ms for r in self.results)[total // 2], 1) if total else 0,
            },
            "per_query": [
                {
                    "query": r.query,
                    "tier": r.tier,
                    "correct": r.correct,
                    "source": r.source,
                    "slm_invoked": r.slm_invoked,
                    "latency_ms": round(r.latency_ms, 1),
                }
                for r in self.results
            ],
            "failures": [
                {
                    "query": r.query,
                    "expected": r.expected_intent,
                    "got": r.actual_intent,
                    "confidence": round(r.confidence, 4),
                    "tier": r.tier,
                    "source": r.source,
                    "notes": r.notes,
                }
                for r in self.results if not r.correct
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n  Results saved to: {filename}")
        return filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NLU v4 Core 17 Eval")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every query result")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to eval/results/")
    parser.add_argument("--tier", "-t", type=str, help="Only run specific tier (clean/messy/adversarial)")
    parser.add_argument("--intent", "-i", type=str, help="Only run specific intent")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--skip-oos", action="store_true", help="Skip OOS queries")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-intent queries")
    parser.add_argument("--skip-boundary", action="store_true", help="Skip boundary queries")
    parser.add_argument(
        "--ablate", type=str, default="",
        help=f"Comma-separated layers to stub out: {','.join(ABLATABLE)}",
    )
    parser.add_argument(
        "--reranker", type=str, default="",
        help="Inject bge-reranker-v2-m3 into SLM slot: 'replace' or 'cascade'",
    )
    parser.add_argument(
        "--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder model name or local path",
    )
    parser.add_argument(
        "--reranker-margin", type=float, default=0.15,
        help="Cascade mode: min softmax margin to answer without the SLM",
    )
    args = parser.parse_args()

    ablate = [a.strip() for a in args.ablate.split(",") if a.strip()]
    unknown = [a for a in ablate if a not in ABLATABLE]
    if unknown:
        parser.error(f"Unknown ablation(s) {unknown}; valid: {list(ABLATABLE)}")
    if args.reranker and args.reranker not in ("replace", "cascade", "rerank"):
        parser.error("--reranker must be 'replace', 'cascade', or 'rerank' "
                     "(rerank = list reordering only; --reranker-margin is the blend alpha)")

    runner = V4EvalRunner(verbose=args.verbose, ablate=ablate,
                          reranker_mode=args.reranker or None,
                          reranker_margin=args.reranker_margin,
                          reranker_model=args.reranker_model)
    runner.nlup()

    # Filter queries
    queries = EVAL_QUERIES
    if args.tier:
        queries = [q for q in queries if q.tier == args.tier]
        print(f"  Filtered to tier={args.tier}: {len(queries)} queries")
    if args.intent:
        queries = [q for q in queries if q.expected_intent == args.intent]
        print(f"  Filtered to intent={args.intent}: {len(queries)} queries")

    # Run
    runner.run_core(queries)

    if not args.skip_oos:
        runner.run_oos(OOS_QUERIES)

    if not args.skip_multi:
        runner.run_multi(MULTI_INTENT_QUERIES)

    if not args.skip_boundary:
        runner.run_boundary(BOUNDARY_QUERIES)

    # Report
    if args.json:
        runner.save_results()
    else:
        runner.print_summary()

    if args.save:
        runner.save_results()

    # Exit code
    total = len(runner.results)
    correct = sum(1 for r in runner.results if r.correct)
    accuracy = correct / total if total else 0
    sys.exit(0 if accuracy >= 0.80 else 1)


if __name__ == "__main__":
    main()
