import json
from pathlib import Path
from difflib import SequenceMatcher

def similar(a, b):
    # similarity metric between strings
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

class EditorInterface:
    def __init__(self, model):
        self.model = model
    def generate_text(self, prompt):
        # generate text from model for a prompt
        return self.model.generate_text(prompt)
    def edit_fact(self, subject, relation, old_obj, new_obj):
        # perform an edit on the model (e.g. ROME or your custom editor)
        self.model.edit_fact(subject, relation, old_obj, new_obj)

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def check_change(editor, prompt, expected_obj, expect_change, threshold=0.6):
    """
    Returns True if change expectation is met:
    - if expect_change==True, model output should contain expected_obj (similar enough)
    - if expect_change==False, model output should NOT change from expected_obj
    """
    output = editor.generate_text(prompt)
    sim = similar(output, expected_obj)
    if expect_change:
        return sim >= threshold, output, sim
    else:
        return sim >= threshold, output, sim

def benchmark(
    editor: EditorInterface,
    edits_path: Path,
    aligned_facts_path: Path,
    distractors_path: Path,
    threshold=0.6
):
    edits = load_jsonl(edits_path)
    aligned_facts = load_jsonl(aligned_facts_path)
    distractors = load_jsonl(distractors_path)

    stats = {
        "total_edits": 0,
        "successful_edits": 0,
        "aligned_facts_total": 0,
        "aligned_facts_unchanged": 0,
        "distractors_total": 0,
        "distractors_changed": 0,
    }

    for edit in edits:
        stats["total_edits"] += 1
        orig = edit["original"]
        subject = orig["subject"]
        relation = orig["relation"]
        old_obj = orig["object"]
        new_obj = edit["target_new_object"]

        # Prepare prompt for generation
        before_prompt = f"{subject} {relation}"

        # Generate before edit
        before_output = editor.generate_text(before_prompt)

        # Apply the edit
        editor.edit_fact(subject, relation, old_obj, new_obj)

        # Generate after edit
        after_output = editor.generate_text(before_prompt)

        sim_before_old = similar(before_output, old_obj)
        sim_after_new = similar(after_output, new_obj)

        edit_success = (sim_after_new >= threshold) and (sim_before_old < threshold)
        if edit_success:
            stats["successful_edits"] += 1

        # Check aligned facts stability: these should NOT change
        for fact in aligned_facts:
            if fact.get("related_to_subject") == subject:
                stats["aligned_facts_total"] += 1
                prompt = fact["prompt"]
                expected_obj = fact.get("expected_object", prompt)  # fallback to prompt if missing
                unchanged, _, sim_val = check_change(editor, prompt, expected_obj, expect_change=False, threshold=threshold)
                if unchanged:
                    stats["aligned_facts_unchanged"] += 1

        # Check distractors: these facts unrelated to subject should remain unchanged
        for fact in distractors:
            stats["distractors_total"] += 1
            prompt = fact["prompt"]
            expected_obj = fact.get("expected_object", prompt)
            unchanged, _, sim_val = check_change(editor, prompt, expected_obj, expect_change=False, threshold=threshold)
            if not unchanged:
                stats["distractors_changed"] += 1

    # Print summary
    print("Benchmark Results:")
    print(f"Total edits attempted: {stats['total_edits']}")
    print(f"Successful edits: {stats['successful_edits']} ({stats['successful_edits'] / max(stats['total_edits'],1)*100:.2f}%)")
    print(f"Aligned facts checked: {stats['aligned_facts_total']}")
    print(f"Aligned facts unchanged: {stats['aligned_facts_unchanged']} ({stats['aligned_facts_unchanged'] / max(stats['aligned_facts_total'],1)*100:.2f}%)")
    print(f"Distractors checked: {stats['distractors_total']}")
    print(f"Distractors changed (undesirable): {stats['distractors_changed']} ({stats['distractors_changed'] / max(stats['distractors_total'],1)*100:.2f}%)")

    return stats

if __name__ == "__main__":
    from editors.knowledge_editor import GPT2ROMEEditor  # assuming your editor class is saved in editor.py

    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark on GPT2ROMEEditor")
    parser.add_argument("--edits", type=Path, required=True, help="Path to edits JSONL file")
    parser.add_argument("--aligned_facts", type=Path, required=True, help="Path to aligned facts JSONL file")
    parser.add_argument("--distractors", type=Path, required=True, help="Path to distractors JSONL file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for success")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2ROMEEditor(device=device)
    editor = EditorInterface(model)

    stats = benchmark(editor, args.edits, args.aligned_facts, args.distractors, threshold=args.threshold)
