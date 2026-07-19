"""Hinglish-TOP public replication — normalizer factorial, deterministic (no SLM).

Usage: python3 eval/run_htop.py --tsv test.tsv --config norm|nonorm --tag lora|base
Saves eval/results/htop_<tag>_<config>.json
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_v4 import NLUPipelineV4
from eval.run_v4_eval import _IdentityNormalizer, _NullSLMReranker


def intent_of(parse):
    m = re.match(r"\[IN:(\w+)", parse or "")
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--config", required=True, choices=["norm", "nonorm"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            it = intent_of(r["cs_parse"]) or intent_of(r["en_parse"])
            if it and r["cs_query"].strip():
                rows.append((r["cs_query"].strip(), it, r["domain"]))
    if args.limit:
        rows = rows[: args.limit]
    print(f"config={args.config} tag={args.tag} n={len(rows)}")

    pipeline = NLUPipelineV4()
    pipeline._slm_reranker = _NullSLMReranker()  # deterministic replication
    if args.config == "nonorm":
        pipeline._normalizer = _IdentityNormalizer()
    pipeline.classify("warmup")

    correct = 0
    pq_log = []
    per_domain = {}
    t0 = time.perf_counter()
    for i, (q, exp, dom) in enumerate(rows):
        res = pipeline.classify(q)
        ok = res.intent == exp
        correct += ok
        pq_log.append({"q": q[:80], "exp": exp, "got": res.intent, "ok": ok})
        d = per_domain.setdefault(dom, [0, 0])
        d[0] += 1
        d[1] += ok
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(rows)} acc={correct/(i+1):.4f}", flush=True)
    dur = time.perf_counter() - t0

    out = {
        "probe": "hinglish_top", "config": args.config, "tag": args.tag,
        "n": len(rows), "accuracy": round(correct / len(rows), 4),
        "per_domain": {k: round(v[1] / v[0], 4) for k, v in per_domain.items()},
        "total_s": round(dur, 1),
        "per_query": pq_log,
    }
    dest = PROJECT_ROOT / "eval" / "results" / f"htop_{args.tag}_{args.config}.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"accuracy={out['accuracy']} ({dur:.0f}s) -> {dest}")


if __name__ == "__main__":
    main()
