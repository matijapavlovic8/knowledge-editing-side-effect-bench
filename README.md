# GPT2 Knowledge Editor

This project is an implementation of the ROME (Rank-One Model Editing) method, designed to enable direct, interpretable edits to the factual knowledge encoded in a large language model—in this case, GPT-2.

## 💡 Project Idea

The goal is to allow users to **surgically edit specific facts** in a pretrained language model (e.g., "Paris is the capital of France" → "Paris is the capital of Italy") without retraining or fine-tuning the model on new data.

This is useful for:
- Rapid knowledge correction without retraining
- Experimenting with internal representations of knowledge
- Studying the effect of localized edits on generative behavior

## 🛠️ Features

- Loads a GPT-2 XL model with HuggingFace Transformers
- Uses ROME to identify and modify factual associations within the model
- Allows generating text before and after an edit to compare behavior
- Supports CPU and GPU execution

## 📦 Dependencies

- PyTorch
- HuggingFace Transformers


## 🔬 Based On

- [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- Original codebase: [https://github.com/kmeng01/rome](https://github.com/kmeng01/rome)

