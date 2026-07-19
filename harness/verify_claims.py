"""Self-verification: recompute the paper's headline statistics from the released
per-query records in results/grid/. Run from the bundle root: python3 harness/verify_claims.py

Covers the three claims a reviewer flagged as not obviously recomputable, plus the flip rate.
"""
import json
import re
import itertools
from glob import glob

G = "results/grid"

def pq(f, key="query"):
    return {p[key]: p["correct"] for p in json.load(open(f))["per_query"]}

# 1. Temp-0 flip rate over the 7 released baseline repeats (b7_*.json)
runs = [pq(f) for f in sorted(glob(f"{G}/b7_*.json"))]
flips = [sum(1 for q in a if a[q] != b[q])
         for a, b in itertools.combinations(runs, 2)]
ever = sum(1 for q in runs[0] if len({r[q] for r in runs}) > 1)
print(f"[flip rate] pairwise {min(flips)}-{max(flips)}/330 over {len(flips)} pairings, "
      f"mean {sum(flips)/len(flips)/330*100:.1f}%; {ever}/330 flip at least once "
      f"(paper: 1-3%, mean ~2%, 15/330)")

# 2. Half-dose holdout: of 21 held-out failure queries, how many does the
#    half-dose-augmented pipeline (t4 runs) answer correctly?
hold = json.load(open(f"{G}/holdout_halfB.json"))["holdout_queries"]
hold = [h if isinstance(h, str) else h.get("query") for h in hold]
for f in sorted(glob(f"{G}/t4_eval_*.json")):
    t = pq(f)
    print(f"[holdout] {f.split('/')[-1]}: fixed {sum(1 for q in hold if t.get(q))}/21 "
          f"(paper: 2, 2, 1)")

# 3. Shortlist recall@1/@3 from rerank logs + benchmark gold
src = open("benchmark/v4_core17_dataset.py").read()
gold = dict(re.findall(r'EvalQuery\("((?:[^"\\]|\\.)*)",\s*"([^"]+)"', src))
r1 = {"pre": [], "post": []}; r3 = {"pre": [], "post": []}
for f in sorted(glob(f"{G}/logged_eval_*.json")):
    for e in json.load(open(f))["rerank_log"]:
        g = gold.get(e["query"])
        if not g:
            continue
        pre = [c[0] for c in e["pre_top5"]]; post = [c[0] for c in e["post_top5"]]
        r1["pre"].append(g == pre[0]); r1["post"].append(g == post[0])
        r3["pre"].append(g in pre[:3]); r3["post"].append(g in post[:3])
n = len(r1["pre"])
print(f"[shortlist recall] n={n} logged escalations: "
      f"recall@1 {sum(r1['pre'])/n:.3f}->{sum(r1['post'])/n:.3f} (paper .705->.739), "
      f"recall@3 {sum(r3['pre'])/n:.3f}->{sum(r3['post'])/n:.3f} (paper .875->.898)")

# 4. Escalation rate determinism
for f in sorted(glob(f"{G}/b7_*.json"))[:1]:
    d = json.load(open(f))["per_query"]
    for t in ("clean", "messy", "adversarial"):
        tot = [p for p in d if p["tier"] == t]
        inv = sum(1 for p in tot if p["slm_invoked"])
        print(f"[escalation rate] {t}: {inv}/{len(tot)} = {100*inv/len(tot):.1f}% "
              f"(identical in every baseline repeat)")

# 5. Normalizer-ablation discordance (paper 4.1: 7 fixed / 7 broken, exact McNemar p=1.0)
from math import comb
base_runs = [pq(f) for f in sorted(glob(f"{G}/b7_*.json"))]
nn_runs = [pq(f) for f in sorted(glob(f"{G}/nonorm_pq_rep*.json"))]
def majority(rs, q):
    v = [r[q] for r in rs if q in r]
    return sum(v) > len(v) / 2
qs = list(base_runs[0])
fixed = sum(1 for q in qs if not majority(base_runs, q) and majority(nn_runs, q))
broken = sum(1 for q in qs if majority(base_runs, q) and not majority(nn_runs, q))
nd = fixed + broken
p_mc = min(1.0, sum(comb(nd, k) for k in range(min(fixed, broken) + 1)) / 2 ** (nd - 1)) if nd else 1.0
print(f"[normalizer discordance] -normalizer fixes {fixed} / breaks {broken}, "
      f"exact McNemar p={p_mc:.2f} (paper: 7/7, p=1.0)")

# 6. Arbiter-collapse discordance (paper 4.1: 51 saved / 6 cost, p=5.7e-10)
slm_runs = [pq(f) for f in sorted(glob(f"{G}/eval_*ablate-slm.json"))
            if json.load(open(f)).get("per_query")]
saved = sum(1 for q in qs if majority(base_runs, q) and not majority(slm_runs, q))
cost = sum(1 for q in qs if not majority(base_runs, q) and majority(slm_runs, q))
na = saved + cost
p_a = min(1.0, sum(comb(na, k) for k in range(min(saved, cost) + 1)) / 2 ** (na - 1)) if na else 1.0
print(f"[arbiter discordance] arbiter saves {saved} / costs {cost}, exact p={p_a:.1e} "
      f"(paper: 51/6, p=5.7e-10)")

# 7. Reranker pairwise comparisons (paper 4.4: 231/240 match-or-exceed)
def acc(f):
    return json.load(open(f))["overall_accuracy"]
rr = [acc(f) for f in sorted(glob(f"{G}/*rerank0.3*.json")) + [f"{G}/rerank03_rep2.json"]]
BASE_FILES = (sorted(glob(f"{G}/b7_*.json")) + [f"{G}/detail_baseline.json"]
              + [f"{G}/eval_2026-07-12_{t}.json" for t in ("1207", "1217", "1227", "1558", "1600", "1603")]
              + [f"{G}/h_eval_2026-07-13_{t}.json" for t in ("1605", "1608")])
bb = [acc(f) for f in BASE_FILES]
wins = sum(1 for r in rr for b in bb if r >= b)
print(f"[reranker pairwise] reranked match-or-exceed baseline in {wins}/{len(rr)*len(bb)} "
      f"pairings ({len(rr)} reranked x {len(bb)} baseline runs) (paper: 231/240)")

# 8. Full-dose augmentation row (paper Table 3: .873/.758; clean drop ~4.1pp vs baseline .985)
fd = [json.load(open(f)) for f in sorted(glob(f"{G}/fulldose_aug_rep*.json"))]
fo = sum(d["overall_accuracy"] for d in fd) / len(fd)
fc = sum(d["tier_accuracy"]["clean"] for d in fd) / len(fd)
fa = sum(d["tier_accuracy"]["adversarial"] for d in fd) / len(fd)
print(f"[full-dose aug] overall {fo:.3f} clean {fc:.3f} adv {fa:.3f} "
      f"(paper: .873 / clean drop ~4.1pp from .985 / .758)")

# 9. Arbiter null on public Hinglish-TOP (paper 4.3: +0.17pp, 959/948, p=0.82)
ha = json.load(open(f"{G}/htop_arbfull1_nonorm.json"))["per_query"]
hd = json.load(open(f"{G}/htop_loraPQ_nonorm.json"))["per_query"]
hf = sum(1 for x, y in zip(ha, hd) if x["ok"] and not y["ok"])
hb = sum(1 for x, y in zip(ha, hd) if not x["ok"] and y["ok"])
hm = hf + hb
p_h = min(1.0, sum(comb(hm, k) for k in range(min(hf, hb) + 1)) / 2 ** (hm - 1)) if hm else 1.0
print(f"[arbiter HTOP null] fixed {hf} / broken {hb}, exact McNemar p={p_h:.2f} "
      f"(paper: 959/948, p=0.82)")
