"""Generate the ta-en (Tanglish) transfer-probe sheet for native-reviewer validation.

Stratified sample of the 330-query benchmark (default 20 clean / 30 messy / 30 adversarial),
with 2 machine-translation candidates per query (llama-3.3-70b via a hosted LLM API) that a native Tamil
reviewer corrects/approves. Output CSV columns:
  tier, intent, hinglish_original, taen_candidate_1, taen_candidate_2,
  REVIEWER_FINAL (blank), REVIEWER_NOTE (blank)
"""
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.v4_core17_dataset import EVAL_QUERIES

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--lang", default="Tamil", help="Tamil|Telugu|Kannada")
_ap.add_argument("--code", default="taen")
_ap.add_argument("--n", type=int, default=80)
_ARGS, _ = _ap.parse_known_args()
_scale = _ARGS.n / 80
N_PER_TIER = {"clean": int(20*_scale), "messy": int(30*_scale), "adversarial": int(30*_scale)}
SEED = 7

PROMPT = f"""You translate Hinglish (romanized Hindi-English) banking queries into natural
romanized {_ARGS.lang}-English code-mix, as a {_ARGS.lang}-speaking bank customer would type
on WhatsApp. Keep English banking loanwords (balance, card, block, EMI, loan, statement)
as-is where a {_ARGS.lang} speaker naturally would. Match the noise level and register of the
original (if it is colloquial/slangy, the translation must be too; if it has shorthand, use
natural shorthand). Do NOT use native script --- Latin-script romanized {_ARGS.lang} only.

Return ONLY a JSON array of exactly 2 alternative translations, no explanation."""


def main():
    rng = random.Random(SEED)
    by_tier = {}
    for q in EVAL_QUERIES:
        by_tier.setdefault(q.tier, []).append(q)
    sample = []
    for tier, n in N_PER_TIER.items():
        sample.extend(rng.sample(by_tier[tier], n))

    from src.llm_client import get_client
    client = get_client()
    model = os.environ.get("DATAGEN_MODEL", "llama-3.3-70b-versatile")
    print(f"model={model} sample={len(sample)}")

    rows = []
    for i, q in enumerate(sample):
        cands = ["", ""]
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": f'Hinglish query: "{q.text}"\nJSON array:'},
                    ],
                    temperature=0.6,
                    max_tokens=300,
                )
                raw = (resp.choices[0].message.content or "").strip()
                start = raw.find("[")
                arr = json.loads(raw[start:raw.rfind("]") + 1])
                if isinstance(arr, list) and len(arr) >= 2:
                    cands = [str(arr[0]).strip(), str(arr[1]).strip()]
                    break
            except Exception as e:
                print(f"  retry {attempt+1} on {q.text[:30]!r}: {e}")
                time.sleep(2)
        rows.append([q.tier, q.expected_intent, q.text, cands[0], cands[1], "", ""])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       f"{_ARGS.code}_transfer_sheet.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "intent", "hinglish_original", "taen_candidate_1",
                    "taen_candidate_2", "REVIEWER_FINAL", "REVIEWER_NOTE"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
