import json
import argparse
from pathlib import Path
import torch
from editors.knowledge_editor import GPT2ROMEEditor

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(data_dir="data", max_edits=10, alpha=0.1, edit_layers=2):
    editor = GPT2ROMEEditor(device="cuda", alpha=alpha, edit_layers=edit_layers)

    edits = load_jsonl(Path(data_dir) / "edits.jsonl")[:max_edits]
    aligned_facts = load_jsonl(Path(data_dir) / "aligned_facts.jsonl")
    distractors = load_jsonl(Path(data_dir) / "distractors.jsonl")

    total, success, aligned_changed, distractors_unchanged = 0, 0, 0, 0

    for edit in edits:
        subj = edit["original"]["subject"]
        rel = edit["original"]["relation"]
        old_obj = edit["original"]["object"]
        new_obj = edit["target_new_object"]
        prompt = edit["prompt"]

        print(f"\n--- Edit #{total+1}: {prompt.strip()} ---")
        print(f"Editing: {old_obj} → {new_obj}")

        # Get model output before edit
        before_output = editor.generate_text(prompt)
        print(f"🔹 Before Edit Output: {before_output}")

        # Apply the edit
        editor.edit_fact(subj, rel, old_obj, new_obj)

        # Get model output after edit
        after_output = editor.generate_text(prompt)
        print(f"🔸 After Edit Output:  {after_output}")

        # Determine whether the new object appears in output
        if new_obj.strip().lower() in after_output.lower():
            print("✅ Edit successful.")
            success += 1
        else:
            print("❌ Edit failed.")

        # Evaluate aligned facts — these SHOULD NOT change
        aligned_related = [fact for fact in aligned_facts if fact["related_to_subject"] == subj]
        for fact in aligned_related:
            aligned_output = editor.generate_text(fact["prompt"])
            if new_obj.strip().lower() in aligned_output.lower():
                aligned_changed += 1
                print(f"Changed aligned fact: {fact['prompt']}")

        # Evaluate distractors — these SHOULD change
        distractor_related = [d for d in distractors if d["subject"] == subj and d["object"] == new_obj]
        for d in distractor_related:
            distractor_output = editor.generate_text(d["prompt"])
            if new_obj.strip().lower() not in distractor_output.lower():
                distractors_unchanged += 1
                print(f"Distractor unchanged: {d['prompt']}")

        total += 1

    print("\n=== Evaluation Summary ===")
    print(f"Edits attempted:        {total}")
    print(f"Successful edits:       {success}")
    print(f"Aligned facts changed:  {aligned_changed}")
    print(f"Distractors unchanged:  {distractors_unchanged}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing edits.jsonl etc.")
    parser.add_argument("--max_edits", type=int, default=10, help="Max number of edits to evaluate")
    parser.add_argument("--alpha", type=float, default=0.4, help="Rank-1 update strength")
    parser.add_argument("--edit_layers", type=int, default=2, help="Number of layers to update")
    args = parser.parse_args()

    main(data_dir=args.data_dir, max_edits=args.max_edits, alpha=args.alpha, edit_layers=args.edit_layers)
