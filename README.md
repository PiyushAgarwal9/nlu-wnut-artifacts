# Where Does Noise Robustness Live? — Artifacts (anonymized review copy)

Artifacts accompanying the W-NUT 2026 submission *"Where Does Noise Robustness Live?
Evidence from Ablation, Role Swaps, and Cross-Lingual Transfer in Code-Mixed Banking NLU."*
The system is referred to as the NLU (system name withheld) for anonymity.

## Contents

- `benchmark/` — BankStress-330, the 330-query tiered benchmark (`v4_core17_dataset.py`): 22 core banking
  intents classified against the full 163-intent space, stratified clean / messy /
  adversarial, with per-query authoring notes. All queries were authored by the research
  team; none derive from real customer communications and no real PII appears anywhere.
- `adjudication/` — the blind adjudication sheet and the final record (38 majority-vote
  failures; 35 gold_ok / 3 gold_wrong) referenced in §3 of the paper.
- `transfer/` — the three native-speaker-validated transfer sets (§4.5): Tamil register T1,
  Tamil register T2 (Gen-Z), Telugu. Columns include the Hinglish original, machine
  candidates, and the reviewer-final query actually evaluated.
- `harness/` — the single-file ablation/reranker harness (`run_v4_eval.py`) that reproduces
  every configuration in the paper against an unmodified pipeline via constructor-level
  substitution (identity normalizer, null arbiter, cross-encoder referee/reranker), plus the
  transfer-probe runner, the Hinglish-TOP replication scripts, the reranker training script,
  and the transfer-sheet generator.
- `results/grid/` — raw per-run result JSONs (per-query records included) for every
  configuration and repeat reported in the paper.
- `reranker/` — config of the fine-tuned cross-encoder reranker. Full weights (~2.1 GB)
  are released at camera-ready (hosting exceeds anonymous-repository limits); the training
  script and pair-construction recipe in `harness/train_xenc_reranker.py` reproduce them.

## Reproducing

The harness runs against the pipeline codebase (released at camera-ready; the harness is
pipeline-agnostic at the seams it ablates). Example invocations:

```
python3 eval/run_v4_eval.py                          # baseline, 330 queries
python3 eval/run_v4_eval.py --ablate normalizer      # layer ablation
python3 eval/run_v4_eval.py --reranker rerank --reranker-model <reranker-weights-dir>
python3 eval/run_taen_transfer.py --csv transfer/tamil_t1_reviewed.csv --config baseline
```

## License

Data (benchmark, transfer sets, adjudication record, result JSONs): CC BY 4.0.
Code (harness, training scripts): Apache 2.0.

## Data dictionary note

In transfer result JSONs the per-query field `taen` holds the evaluated transfer-language
query regardless of language (a historical field name from the first Tamil probe); `hinglish`
holds the source query it was translated from. File-name prefixes (`tamil_t1`, `tamil_t2_genz`,
`telugu`) and the top-level `probe` field identify the set.

## Result-file eras and known artifacts

- `xenc_replace_v1_gatebug_equals_noarbiter.json`: first cross-encoder-as-decider attempt.
  A confidence-gating bug caused every cross-encoder verdict to be rejected downstream, so the
  run degenerates to the --arbiter configuration (.761/.545). Kept for transparency;
  superseded by `xenc_replace_v2.json` (zero-shot, .530) and
  `eval_2026-07-12_1555_xenc-replace0.15.json` (fine-tuned, .461), which are the two rows in
  the paper's Table 3.
- A mid-study harness patch (per-query logging + ablation stubs) shifted exactly one
  --arbiter query: `eval_2026-07-12_12xx_ablate-slm` files score .7606, later ones .7576.
  Deterministic bit-identity holds within each harness era.
- `eval_2026-07-12_1[56]xx` baseline files carry an older `slm_invoked` logging convention;
  escalation analysis should use `b7_*.json` / `detail_baseline.json`.
- `harness/verify_claims.py` recomputes the flip rate, half-dose holdout counts, shortlist
  recall, and escalation rates from the released records alone.
- `benchmark/v4_core17_dataset.py`: the filename reflects the original 17 action intents; the
  file (and paper) evaluates 22 core intents (17 action + 4 explore + 1 continuation).
- `htop_base_{norm,nonorm}.json`: base-encoder (no LoRA) Hinglish-TOP runs backing the
  paper's ".603 both" cell.
