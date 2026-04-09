import os
import json
import random

from datasets import load_dataset

NUM_DOCS = 10  # Change to 30 or 75 to scale up — nothing else changes

SHORT_MAX = 400   # words
MEDIUM_MAX = 700  # words
TRUNCATE_WORDS = 1200


def word_count(text):
    return len(text.split())


def length_tier(text):
    wc = word_count(text)
    if wc < SHORT_MAX:
        return "short"
    elif wc < MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def truncate(text, max_words=TRUNCATE_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def main():
    os.makedirs(".tmp", exist_ok=True)

    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    # Bucket documents by length tier
    buckets = {"short": [], "medium": [], "long": []}
    for row in dataset:
        tier = length_tier(row["article"])
        buckets[tier].append(row)

    print(f"Tier counts before sampling — short: {len(buckets['short'])}, "
          f"medium: {len(buckets['medium'])}, long: {len(buckets['long'])}")

    # Determine per-tier sample sizes
    base = NUM_DOCS // 3
    remainder = NUM_DOCS - base * 3  # put remainder in medium
    per_tier = {"short": base, "medium": base + remainder, "long": base}

    random.seed(42)
    sampled = []
    for tier, count in per_tier.items():
        pool = buckets[tier]
        if len(pool) < count:
            raise ValueError(f"Not enough {tier} docs: need {count}, have {len(pool)}")
        sampled.extend(random.sample(pool, count))

    # Assign doc_ids and write
    output_path = ".tmp/dataset.jsonl"
    with open(output_path, "w") as f:
        for idx, row in enumerate(sampled):
            doc_id = f"cnn_{idx:04d}"
            record = {
                "doc_id": doc_id,
                "document": truncate(row["article"]),
                "reference_summary": row["highlights"],
                "length_tier": length_tier(row["article"]),  # tier based on full doc
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(sampled)} documents to {output_path}")
    tiers_written = [length_tier(r["article"]) for r in sampled]
    for t in ("short", "medium", "long"):
        print(f"  {t}: {tiers_written.count(t)}")


if __name__ == "__main__":
    main()
