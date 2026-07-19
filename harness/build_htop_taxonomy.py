"""Build a taxonomy_v4-style YAML from Hinglish-TOP train split (public replication)."""
import csv
import re
import sys
from collections import defaultdict

import yaml

TRAIN = sys.argv[1] if len(sys.argv) > 1 else "htop/Dataset/Human Annotated Data/train.tsv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "config/taxonomy_v4.yaml"
MAX_EX = 50

def intent_of(parse):
    m = re.match(r"\[IN:(\w+)", parse or "")
    return m.group(1) if m else None

by_domain = defaultdict(lambda: defaultdict(list))
with open(TRAIN) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        it = intent_of(r["cs_parse"]) or intent_of(r["en_parse"])
        if not it:
            continue
        q = r["cs_query"].strip()
        if q and len(by_domain[r["domain"]][it]) < MAX_EX:
            by_domain[r["domain"]][it].append(q)

domains = {}
n_int = n_ex = 0
for dom, intents in by_domain.items():
    dcfg = {"intents": {}}
    for it, exs in intents.items():
        n_int += 1
        n_ex += len(exs)
        dcfg["intents"][it] = {
            "description": it.replace("_", " ").lower(),
            "mode": "conversation",
            "risk": "low",
            "examples": exs,
        }
    domains[dom] = dcfg

yaml.safe_dump({"domains": domains, "global": {}}, open(OUT, "w"),
               allow_unicode=True, width=1000)

# Index builder reads data/utterances_v4/<intent>.yaml (plain list) — write them
from pathlib import Path
import shutil
utt_dir = Path(OUT).resolve().parent.parent / "data" / "utterances_v4"
if utt_dir.exists():
    shutil.rmtree(utt_dir)
utt_dir.mkdir(parents=True)
for dom, intents in by_domain.items():
    for it, exs in intents.items():
        yaml.safe_dump(list(exs), open(utt_dir / f"{it}.yaml", "w"),
                       allow_unicode=True, width=1000)
print(f"domains={len(domains)} intents={n_int} examples={n_ex} -> {OUT} + {utt_dir}")
