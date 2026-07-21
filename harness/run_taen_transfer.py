"""ta-en (Tanglish) transfer probe — W-NUT §4.5.
# REFERENCE RUNNER (not standalone). Assumes the original repo layout (eval/, src/)
# and the pipeline source + indices, released at camera-ready. This bundle is a
# result-verification package; use harness/verify_claims.py to recompute headline stats.

Runs reviewer-validated Tanglish queries through the pipeline under 4 configs:
  baseline | -normalizer | -slm | +FT-rerank (alpha 0.3)
Usage:
  python3 eval/run_taen_transfer.py --csv taen_reviewed.csv --config baseline --rep 1 \
      [--reranker-model DIR]
Saves eval/results/taen_<config>_rep<k>.json with per-query records.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_v4 import NLUPipelineV4
from eval.run_v4_eval import (
    _IdentityNormalizer, _NullSLMReranker, _RerankingRetriever,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True,
                    choices=["baseline", "nonorm", "noslm", "ftrerank",
                             "norm_agnostic", "norm_hinglish"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--reranker-model", default="")
    ap.add_argument("--out-prefix", default="taen")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv))
            if (r.get("REVIEWER_FINAL") or "").strip()]
    print(f"config={args.config} rep={args.rep} queries={len(rows)}")

    pipeline = NLUPipelineV4()
    if args.config == "nonorm":
        pipeline._normalizer = _IdentityNormalizer()
    elif args.config == "norm_agnostic":
        # keep language-agnostic stages; disable Hinglish-specific rewriting
        n = pipeline._get_normalizer()
        n._phonetic_patterns = []
        n._hinglish_std = {}
    elif args.config == "norm_hinglish":
        # keep ONLY Hinglish-specific rewriting; disable abbreviations
        n = pipeline._get_normalizer()
        n._abbreviations = {}
    elif args.config == "noslm":
        pipeline._slm_reranker = _NullSLMReranker()
    elif args.config == "ftrerank":
        inner = pipeline._get_hybrid_retriever()
        pipeline._hybrid_retriever = _RerankingRetriever(
            inner, model_name=args.reranker_model or "BAAI/bge-reranker-v2-m3",
            descs=pipeline._taxonomy.intent_descriptions, alpha=0.3,
        )
    pipeline.classify("warmup query")

    per_query = []
    tiers = {}
    for r in rows:
        q = r["REVIEWER_FINAL"].strip()
        t0 = time.perf_counter()
        res = pipeline.classify(q)
        lat = (time.perf_counter() - t0) * 1000
        ok = res.intent == r["intent"]
        tiers.setdefault(r["tier"], [0, 0])
        tiers[r["tier"]][0] += 1
        tiers[r["tier"]][1] += int(ok)
        per_query.append({
            "tier": r["tier"], "taen": q, "hinglish": r["hinglish_original"],
            "expected": r["intent"], "got": res.intent, "correct": ok,
            "confidence": round(res.confidence, 4), "latency_ms": round(lat, 1),
        })

    total = len(per_query)
    correct = sum(1 for p in per_query if p["correct"])
    out = {
        "probe": "taen_transfer", "config": args.config, "rep": args.rep,
        "overall_accuracy": round(correct / total, 4),
        "tier_accuracy": {k: round(v[1] / v[0], 4) for k, v in tiers.items()},
        "tier_counts": {k: v[0] for k, v in tiers.items()},
        "per_query": per_query,
    }
    dest = PROJECT_ROOT / "eval" / "results" / f"{args.out_prefix}_{args.config}_rep{args.rep}.json"
    json.dump(out, open(dest, "w"), ensure_ascii=False, indent=1)
    print(f"overall={out['overall_accuracy']} tiers={out['tier_accuracy']}")
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
