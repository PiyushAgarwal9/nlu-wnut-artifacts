# Table/Figure -> result-file manifest

| Paper element | Backing files in results/grid/ |
|---|---|
| Table 1 (layer ablation) baseline row | `b7_*.json` (7 repeats, per-query) + `eval_2026-07-12_1207/1217/1227.json` (aggregates) |
| Table 1 --normalizer row | `nonorm_pq_rep1-3.json` (per-query) + `eval_*_ablate-normalizer.json` (aggregates) |
| Table 1 --rules row | `eval_*_ablate-rules.json` |
| Table 1 --arbiter row + 4.1 discordance 51/6 | `eval_2026-07-13_1951/1952_ablate-slm.json`, `eval_2026-07-14_1828_ablate-slm.json` (per-query); `eval_2026-07-12_*_ablate-slm.json` (earlier harness era, aggregates) |
| Table 2 (factorial, base e5) | `t3_eval_*.json` |
| Hinglish-TOP replication (4.1) | `htop_loraPQ_norm.json`, `htop_loraPQ_nonorm.json` (per-query); base-encoder cell `htop_base_{norm,nonorm}.json` (aggregates) |
| Arbiter null on Hinglish-TOP (4.3) | `htop_arbfull1_nonorm.json` (with-arbiter, per-query) vs `htop_loraPQ_nonorm.json` (deterministic) |
| Fig. 2 escalation rate/precision | rates: any `b7_*.json` / `detail_baseline.json`; precision: `b7_*` x per-query ablate-slm files via `harness/esc_precision.py` |
| Table 3 arbiter substitutes | zero-shot decider `xenc_replace_v2.json`; FT decider `eval_2026-07-12_1555_xenc-replace0.15.json`; 8B arbiter `8b_eval_*.json`; (superseded: `xenc_replace_v1_gatebug_equals_noarbiter.json`) |
| Table 3 augmentation rows | full dose `fulldose_aug_rep1-3.json` (augmented-taxonomy runs, per-query); half dose `t4_eval_*.json` + `holdout_halfB.json` + `eval_split_halfA.json`. (`t2_eval_*.json` are frozen-checkout verification runs, previously mislabeled as full-dose.) |
| Table 4 zero-shot row (.929/.779/.890) | 3 zero-shot runs: `eval_2026-07-12_1631_xenc-rerank0.3.json`, `eval_2026-07-12_1648_xenc-rerank0.3.json`, `rerank03_rep2.json` |
| Table 4 .906 (3 FT table runs) | `eval_2026-07-12_1651/1659/1702_xenc-rerank0.3.json` (mean .906); the other 12 rerank0.3 runs (15 total, all-run mean .901) back only the 231/240 pairwise inference, not the Table 4 cell |
| Table 4 (FT rerank) + 4.4 stats | `eval_*_xenc-rerank0.3.json`, `h_eval_*_xenc-rerank0.3.json`, `rerank03_rep2.json` (15 reranked runs; pairwise claim uses all 16 baseline-run files) vs `b7_*` + baseline aggregates; recall: `logged_eval_*_xenc-rerank0.3.json` |
| Table 5 (pareto/latency) | latency fields of the files above |
| Table 6 transfer ta-T1 | `tamil_t1_{baseline,nonorm,noslm,ftrerank}_rep*.json`; stage-wise: `tamil_t1_norm_{agnostic,hinglish}_rep*.json` |
| Table 6 transfer ta-T2 genz | `tamil_t2_genz_{baseline,nonorm}_rep*.json` |
| Table 6 transfer te | `telugu_{baseline,nonorm,noslm,ftrerank}_rep*.json` |
| Residual-failure taxonomy / 38 failures | 4 adjudication-designated repeats: `b7_1711.json`, `b7_1720.json`, `b7_1723.json`, `b7_1726.json` (majority = fail in >=2 of 4 -> 38; ever-failing 43). All 7 released b7 repeats give 35 majority / 46 ever-failing. |
| 5 adjudication | `../adjudication/adjudication_record_FINAL.csv` |
