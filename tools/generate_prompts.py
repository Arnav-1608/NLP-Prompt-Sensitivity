import os
import json

DATASET_PATH = ".tmp/dataset.jsonl"
OUTPUT_PATH = ".tmp/all_prompts.jsonl"

REPS = 3

# Two hardcoded few-shot examples (topically neutral, <100 words each)
FEW_SHOT_EXAMPLES = [
    {
        "article": (
            "Scientists at the university have developed a new battery technology "
            "that charges in under five minutes. The research team tested the prototype "
            "on electric vehicles and found it outperforms current lithium-ion cells. "
            "The technology could be commercially available within three years."
        ),
        "summary": "Researchers have created a fast-charging battery that could revolutionise electric vehicles within three years.",
    },
    {
        "article": (
            "The city council voted on Tuesday to close three public libraries due to "
            "budget shortfalls. Residents protested outside the council chambers, holding "
            "signs and chanting. The closures are set to take effect next month unless "
            "alternative funding is found."
        ),
        "summary": "Three public libraries face imminent closure after the city council approved budget cuts, prompting protests from residents.",
    },
]


def build_few_shot_prefix():
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Article: {ex['article']}\nSummary: {ex['summary']}")
    return "\n\n".join(lines)


FEW_SHOT_PREFIX = build_few_shot_prefix()


def make_prompts(doc):
    document = doc["document"]
    return {
        "zero_shot": (
            f"Summarize the following article in 2-3 sentences.\n\n"
            f"Article: {document}\n\nSummary:"
        ),
        "role_primed": (
            f"You are an expert summarizer. Summarize the following article in 2-3 sentences.\n\n"
            f"Article: {document}\n\nSummary:"
        ),
        "few_shot": (
            f"{FEW_SHOT_PREFIX}\n\n"
            f"Article: {document}\n\nSummary:"
        ),
        "chain_of_thought": (
            f"Read the article carefully. First, identify the main points. "
            f"Then write a 2-3 sentence summary.\n\n"
            f"Article: {document}\n\nLet me identify the key points:\n"
        ),
        "perturbation_1": (
            f"Please provide a 2-3 sentence summary of the article below.\n\n"
            f"Article: {document}\n\nSummary:"
        ),
        "perturbation_2": (
            f"Write a brief summary (2-3 sentences) for the following article.\n\n"
            f"Article: {document}\n\nSummary:"
        ),
        "perturbation_3": (
            f"Condense the article below into 2-3 sentences.\n\n"
            f"Article: {document}\n\nSummary:"
        ),
    }


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run prepare_dataset.py first."
        )

    docs = []
    with open(DATASET_PATH) as f:
        for line in f:
            docs.append(json.loads(line))

    rows = []
    for doc in docs:
        prompts = make_prompts(doc)
        for style, prompt_text in prompts.items():
            for rep in range(1, REPS + 1):
                prompt_id = f"{style}__{doc['doc_id']}__rep{rep}"
                rows.append({
                    "prompt_id": prompt_id,
                    "doc_id": doc["doc_id"],
                    "prompt_style": style,
                    "rep": rep,
                    "prompt_text": prompt_text,
                    "reference_summary": doc["reference_summary"],
                    "source_document": doc["document"],
                })

    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    expected = len(docs) * 7 * REPS
    print(f"Saved {len(rows)} prompt rows to {OUTPUT_PATH} (expected {expected})")
    styles_seen = sorted(set(r["prompt_style"] for r in rows))
    print(f"Prompt styles: {styles_seen}")


if __name__ == "__main__":
    main()
