"""Escalation precision with repeat-based ranges.

Pair each per-query baseline run with each per-query -arbiter run. Per tier:
  escalated  = queries with slm_invoked=True in the baseline run
  helped     = escalated & correct in baseline & wrong in -arbiter
  hurt       = escalated & wrong in baseline & correct in -arbiter
  precision  = helped / escalated   (paper Fig. 2: ~25/16/11 adv/messy/clean)
Reports mean and min-max across all pairings.
"""
import json
import sys
from glob import glob

base_files = sys.argv[1].split(",")
noslm_files = sys.argv[2].split(",")

def per_query(f):
    return {p["query"]: p for p in json.load(open(f))["per_query"]}

tiers = ["clean", "messy", "adversarial"]
prec = {t: [] for t in tiers}
rate = {t: [] for t in tiers}
for bf in base_files:
    b = per_query(bf)
    for nf in noslm_files:
        n = per_query(nf)
        for t in tiers:
            esc = [q for q, p in b.items() if p["tier"] == t and p["slm_invoked"] and q in n]
            total_t = sum(1 for p in b.values() if p["tier"] == t)
            helped = sum(1 for q in esc if b[q]["correct"] and not n[q]["correct"])
            prec[t].append(100 * helped / len(esc))
            rate[t].append(100 * len(esc) / total_t)

for t in tiers:
    p, r = prec[t], rate[t]
    print(f"{t:12s} rate {sum(r)/len(r):5.1f}% [{min(r):.1f},{max(r):.1f}]  "
          f"precision {sum(p)/len(p):5.1f}% [{min(p):.1f},{max(p):.1f}]  ({len(p)} pairings)")
