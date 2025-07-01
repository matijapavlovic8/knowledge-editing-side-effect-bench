import json
import pprint
from datasets import load_dataset
from pathlib import Path

output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("azhx/counterfact", split="train")
print(f"Loaded {len(dataset)} examples from CounterFact.")

for i in range(2):
    print(f"\n--- Example #{i} ---")
    pprint.pprint(dataset[i], compact=False, width=120)

MAX_EXAMPLES = 1000
dataset = dataset.select(range(min(MAX_EXAMPLES, len(dataset))))

triples = []
edits = []
aligned_facts = []
distractors = []

for example in dataset:
    rw = example["requested_rewrite"]
    subject = rw["subject"]
    relation = rw["relation_id"] or "unknown"
    object_true = rw["target_true"]["str"]
    object_new = rw["target_new"]["str"]

    # Add to triples
    triples.append({
        "subject": subject,
        "relation": relation,
        "object": object_true,
        "source": "CounterFact"
    })

    # Use first generation_prompt as the natural-language prompt for edits
    gen_prompts = example.get("generation_prompts", [])
    if gen_prompts:
        nl_prompt = gen_prompts[0]
    else:
        # Fallback: simple template
        nl_prompt = f"{subject} {relation}"

    edits.append({
        "original": {
            "subject": subject,
            "relation": relation,
            "object": object_true
        },
        "target_new_object": object_new,
        "prompt": nl_prompt,
        "category": "generation"
    })

    # Aligned facts: neighborhood and attribute prompts
    for neighborhood_prompt in example.get("neighborhood_prompts", []):
        aligned_facts.append({
            "prompt": neighborhood_prompt,
            "related_to_subject": subject,
            "expected_change": False,
            "category": "neighborhood"
        })

    for attribute_prompt in example.get("attribute_prompts", []):
        aligned_facts.append({
            "prompt": attribute_prompt,
            "related_to_subject": subject,
            "expected_change": False,
            "category": "attribute"
        })

    # Distractor prompts: generation and paraphrase templates
    for prompt_str in gen_prompts:
        distractors.append({
            "subject": subject,
            "relation": relation,
            "object": object_new,
            "prompt": prompt_str,
            "expected_change": True,
            "category": "generation"
        })

    for paraphrase_prompt in example.get("paraphrase_prompts", []):
        distractors.append({
            "subject": subject,
            "relation": relation,
            "object": object_new,
            "prompt": paraphrase_prompt,
            "expected_change": True,
            "category": "paraphrase"
        })


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Write all JSONL outputs
write_jsonl(output_dir / "triples.jsonl", triples)
write_jsonl(output_dir / "edits.jsonl", edits)
write_jsonl(output_dir / "aligned_facts.jsonl", aligned_facts)
write_jsonl(output_dir / "distractors.jsonl", distractors)

print(f"Saved {len(triples)} triples, {len(edits)} edits, {len(aligned_facts)} aligned facts, and {len(distractors)} distractors.")
