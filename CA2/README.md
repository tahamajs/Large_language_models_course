# CA2 — Taha Majlesi (Notebook)

This folder contains `CA2_Taha_Majlesi_810101504.ipynb`, the CA2 assignment notebook for the Large Language Models course. This README gives reproducible setup instructions, explains the notebook structure, how to run training/evaluation, and short troubleshooting tips.

---

## Files

- `CA2_Taha_Majlesi_810101504.ipynb` — main notebook.
- `README.md` — this file.

---

## Goal

The notebook demonstrates dataset preparation, model/tokenizer setup, training (optionally with LoRA/PEFT), and evaluation. It includes visualization cells and diagnostic outputs.

---

## Environment & Requirements

Recommended: Python 3.9+ in a virtual environment.

Install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn pandas seaborn matplotlib
pip install accelerate peft  # only if you use LoRA/PEFT
```

Notes:

- Replace `torch` installation command with a specific CUDA-compatible wheel if you have a GPU.
- If running on an M1/M2 Mac, ensure the PyTorch wheel supports your architecture or use `conda`.

---

## Quick start — run order

1. Open the notebook in JupyterLab or Jupyter Notebook.
2. Run the top setup cell (imports, device selection, helper functions). It creates `run_inference` and `compute_and_save_metrics` helpers.
3. Run the data-loading cell. Options:
   - Set `load_hf_dataset = True` and `hf_dataset_name` if you want to load a public Hugging Face dataset.
   - Or set `data_path` to point to a CSV file with a text column and a label column.
4. Run the model/tokenizer setup cell. Set `model_name_or_path` and optionally `num_labels`.
5. Run training cells (if you intend to train). Adjust training arguments as necessary.
6. Run the evaluation cell to compute metrics and save misclassified sample CSV.

---

## Data format

- The data-loading cell attempts to auto-detect text columns from `['text', 'sentence', 'content']` and infer a label column (prefers `label`, otherwise last column).
- If your file uses different column names, edit the cell to specify `text_col` and `label_col` explicitly.

---

## Evaluation

- After inference, the evaluation helper prints accuracy and macro F1, shows a per-class report, draws a confusion matrix, and saves `misclassified_samples.csv` if the dataset provides text examples.
- Use the CSV to manually inspect problematic predictions and identify label noise or ambiguous examples.

---

## Troubleshooting

- Import errors: activate the correct virtualenv where dependencies are installed and restart the kernel.
- Out-of-memory: reduce batch size, max sequence length, or enable 8-bit optimizers if supported.
- Tokenization mismatch: ensure the same tokenizer is used for both training and inference.

---

## Notebook Structure

The notebook is divided into the following major sections (based on cell groupings and markdown headers). Each section includes code cells for implementation and markdown cells for explanations.

1. **Introduction and Setup (Cells 1-10)**  
   Markdown cells introducing the assignment, imports, and initial setup. Includes library imports and device configuration.

2. **Data Exploration and Preparation (Cells 11-30)**  
   Cells for loading and exploring datasets, including CSV or Hugging Face datasets. Covers data preprocessing, tokenization, and basic statistics.

3. **Model and Tokenizer Setup (Cells 31-50)**  
   Code for loading pre-trained models and tokenizers from Hugging Face. Includes configuration for fine-tuning and PEFT/LoRA.

4. **Training and Fine-Tuning (Cells 51-100)**  
   Implementation of training loops, loss functions, and optimization. Covers LoRA configuration, training arguments, and checkpoint saving.

5. **Evaluation and Metrics (Cells 101-130)**  
   Cells for running inference, computing accuracy/F1, confusion matrices, and per-class reports. Includes visualization of results.

6. **Additional Experiments and Visualizations (Cells 131-149)**  
   Optional cells for hyperparameter sweeps, data augmentation, or advanced visualizations like plots and charts.

Each section has inline comments and markdown explanations. Run cells sequentially to avoid dependency issues.

---

## Suggested experiments

- Try small LoRA sweeps (`r` in {4,8,16}, `alpha` in {8,16,32}).
- If class imbalance exists, test class-weighted losses or simple oversampling.
- Use temperature scaling on validation set probabilities to calibrate scores.

---

If you'd like, I can:

- Run the evaluation here and attach the `misclassified_samples.csv` and confusion matrix.
- Add a `requirements.txt` with exact versions used in this workspace.
- Add sample `data/*.csv` loader examples or a small script to convert CSV → HF Dataset.

Tell me which of these you'd like next.
