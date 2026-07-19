"""Fine-tune bge-reranker-v2-m3 for intent reranking (W-NUT Track 2b).

Builds (query, intent_text) pairs from taxonomy_v4.yaml:
  positive  = (example utterance, its own intent text)     label 1
  hard neg  = (example, sibling intent in same domain)     label 0
  rand neg  = (example, random other-domain intent)        label 0
intent_text format matches eval-time scoring exactly:
  "<description>. e.g. <one example>"

Train on GPU. Output: models/xenc_intent_reranker/
Usage:
  python3 scripts/train_xenc_reranker.py --taxonomy config/taxonomy_v4.yaml \
      --epochs 1 --batch 24 --out models/xenc_intent_reranker
"""
import argparse
import random
import yaml
from pathlib import Path


def load_taxonomy(path):
    raw = yaml.safe_load(open(path))
    intents = {}  # name -> {description, examples, domain}
    for dom, dcfg in (raw.get("domains") or {}).items():
        for name, icfg in (dcfg.get("intents") or {}).items():
            ex = icfg.get("examples") or []
            desc = icfg.get("description") or name.replace("_", " ")
            if ex:
                intents[name] = {"desc": desc, "examples": ex, "domain": dom}
    return intents


def intent_text(meta):
    ex = meta["examples"][0] if meta["examples"] else ""
    return f"{meta['desc']}. e.g. {ex}" if ex else meta["desc"]


def build_pairs(intents, hard_neg=3, rand_neg=2, seed=13):
    rng = random.Random(seed)
    names = list(intents)
    by_domain = {}
    for n, m in intents.items():
        by_domain.setdefault(m["domain"], []).append(n)
    samples = []
    for name, meta in intents.items():
        siblings = [s for s in by_domain[meta["domain"]] if s != name]
        others = [o for o in names if intents[o]["domain"] != meta["domain"]]
        for q in meta["examples"]:
            samples.append((q, intent_text(meta), 1.0))
            for s in rng.sample(siblings, min(hard_neg, len(siblings))):
                samples.append((q, intent_text(intents[s]), 0.0))
            for o in rng.sample(others, min(rand_neg, len(others))):
                samples.append((q, intent_text(intents[o]), 0.0))
    rng.shuffle(samples)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="config/taxonomy_v4.yaml")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="models/xenc_intent_reranker")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    intents = load_taxonomy(args.taxonomy)
    samples = build_pairs(intents)
    if args.max_pairs:
        samples = samples[: args.max_pairs]
    n_pos = sum(1 for s in samples if s[2] == 1.0)
    print(f"intents={len(intents)} pairs={len(samples)} (pos={n_pos}, neg={len(samples)-n_pos})")

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    train = [InputExample(texts=[q, t], label=l) for q, t, l in samples]
    model = CrossEncoder(args.model, num_labels=1, max_length=256)
    dl = DataLoader(train, shuffle=True, batch_size=args.batch)
    warmup = max(10, int(len(dl) * 0.05))
    print(f"steps/epoch={len(dl)} warmup={warmup}")
    model.fit(
        train_dataloader=dl,
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.out,
        use_amp=True,
    )
    model.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
