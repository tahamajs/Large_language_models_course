# Large Language Models Course Assignments

This repository contains Jupyter notebooks for assignments in the Large Language Models (LLM) course. Each assignment explores different aspects of LLMs, including tokenization, fine-tuning, evaluation, RAG, quantization, and more.

## Table of Contents

- [CA1: LLM Practical Assignment](#ca1-llm-practical-assignment)
- [CA2: LLM Assignment](#ca2-llm-assignment)
- [CA3: LLM Homework 3](#ca3-llm-homework-3)
- [CA4: Quantization and Self-Explanations](#ca4-quantization-and-self-explanations)
- [Papers](#papers)
- [Setup and Requirements](#setup-and-requirements)
- [General Instructions](#general-instructions)

## CA1: LLM Practical Assignment

**Notebook:** `Files/CAs/CA1/code/LLM_CA1_final.ipynb`

This assignment explores various aspects of Large Language Models, including tokenization, model generation, fine-tuning techniques, and performance evaluation. The project demonstrates hands-on experience with modern LLM techniques using the Llama-3.2-1B models and the emotion detection dataset.

### Project Structure

```
CA1/
├── code/
│   ├── LLM_CA1_final.ipynb    # Main notebook with all experiments
│   └── README.md              # Detailed README for CA1
├── papers/
│   ├── 2021.acl-long.353.pdf
│   ├── 2021.emnlp-main.243.pdf
│   └── 2103.10385v2.pdf
└── [other directories...]
```

### Prerequisites

- Python 3.8 or higher
- Access to Hugging Face models (specifically `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-1B-Instruct`, `mistralai/Mistral-7B-v0.1`, `microsoft/Phi-4-mini-instruct`)
- Sufficient GPU memory (recommended: at least 8GB VRAM for full fine-tuning, less for LoRA)
- Hugging Face account with model access tokens

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv ca1_env
   source ca1_env/bin/activate  # On Windows: ca1_env\Scripts\activate
   ```
2. Install the required packages:
   ```bash
   pip install torch transformers datasets scikit-learn pandas matplotlib seaborn peft accelerate
   ```
3. Set up Hugging Face authentication with access tokens for gated models.

### Usage

1. Launch Jupyter Notebook and open `LLM_CA1_final.ipynb`
2. Follow the cells in order: setup, tokenization experiments, model comparison, LoRA fine-tuning, evaluation.

### Key Experiments

The notebook covers comprehensive exploration of LLMs:

#### Q1: First Steps (25 pts)

- **Q1.1: Readable Model Generation**: Understanding token outputs vs. human-readable text, using tokenizer decode.
- **Q1.2: Generation Function**: Creating reusable text generation with parameters like temperature, top-k, top-p.
- **Q1.3: Comparing Tokenizers**: Analysis across model families, including Persian text handling.
- **Q1.4: Base vs. Instruction-Tuned Models**: Performance comparison, response styles, temperature impact.
- **Q1.5: Chat Templates**: Implementing ChatML format, multi-turn conversations.

#### Q2: Fine-Tuning Using LoRA (75 pts)

- **Q2.0: Utilities**: Stratified sampling for dataset balancing.
- **Q2.1: Data Preparation**: Conversation formatting, tokenization with masking.
- **Q2.2: LoRA Configuration**: Rank, alpha, target modules, dropout exploration.
- **Q2.3: Training Callbacks**: Custom callbacks, early stopping, monitoring.
- **Q2.4: Training Arguments**: Hyperparameter optimization, scheduling.
- **Q2.5: Memory Analysis**: Full fine-tuning vs. LoRA memory requirements.
- **Q2.6: Training Execution**: LoRA adapter training, model saving.
- **Q2.7-2.8: Alternative Methods**: IA3, soft prompt techniques (Prompt Tuning, Prefix Tuning).
- **Q2.9: Output Generation**: Inference pipeline for emotion detection.
- **Q2.10: Performance Visualization**: Accuracy, F1-score metrics, comparative analysis.

### Parts:

1. **Introduction and Setup** (Cells 1-10): Markdown introductions, library imports, device setup.
2. **Q1: Tokenization & Generation** (Cells 11-50): Tokenization, generation parameters, chat templates.
3. **Q2: Dataset & Fine-Tuning** (Cells 51-100): Data loading, LoRA setup, training loops, evaluation.
4. **Evaluation & Analysis** (Cells 101-162): Metrics computation, confusion matrices, ARTS analysis.
5. **Instructor Answers** (Cells 163-170): Consolidated guidance for Q1/Q2.

### Results and Findings

- Tokenization: Llama models handle Persian text better than Mistral and Phi.
- Model Behavior: Instruction-tuned models are more focused and concise.
- Fine-Tuning Efficiency: LoRA enables fine-tuning with ~0.9% of parameters.
- Performance Gains: Fine-tuned models show substantial improvements on emotion detection.

### Setup:

- Python 3.9+, install `transformers`, `datasets`, `peft`, `torch`, etc.
- Run cells sequentially; evaluation produces `misclassified_samples.csv`.

## CA2: LLM Assignment

**Notebook:** `Files/CAs/CA2/code/CA2_Taha_Majlesi_810101504.ipynb`

This assignment demonstrates dataset preparation, model/tokenizer setup, training (optionally with LoRA/PEFT), and evaluation for a classification task. It includes visualization cells and diagnostic outputs.

### Project Structure

```
CA2/
├── base_code/
│   └── LLM_CA2.ipynb          # Base code notebook
├── code/
│   └── CA2_Taha_Majlesi_810101504.ipynb  # Main submission notebook
├── papers/
│   ├── 2024.naacl-long.228.pdf
│   └── 2201.11903v6.pdf
└── README.md                  # Detailed README for CA2
```

### Environment & Requirements

- Python 3.9+ in a virtual environment.
- Install dependencies:
  ```bash
  pip install torch transformers datasets scikit-learn pandas seaborn matplotlib accelerate peft
  ```
- For GPU: Use CUDA-compatible PyTorch wheel.

### Quick Start — Run Order

1. Open the notebook in JupyterLab or Jupyter Notebook.
2. Run the top setup cell (imports, device selection, helper functions).
3. Run the data-loading cell: Set `load_hf_dataset = True` and `hf_dataset_name` for Hugging Face datasets, or `data_path` for CSV files.
4. Run the model/tokenizer setup cell: Set `model_name_or_path` and optionally `num_labels`.
5. Run training cells if training is needed. Adjust training arguments.
6. Run the evaluation cell to compute metrics and save `misclassified_samples.csv`.

### Data Format

- Auto-detects text columns from `['text', 'sentence', 'content']` and label column (prefers `label`, otherwise last column).
- Edit cell to specify `text_col` and `label_col` if needed.

### Evaluation

- Prints accuracy and macro F1, shows per-class report, confusion matrix.
- Saves `misclassified_samples.csv` for manual inspection.

### Troubleshooting

- Import errors: Activate correct virtualenv and restart kernel.
- Out-of-memory: Reduce batch size, max sequence length, or enable 8-bit optimizers.
- Tokenization mismatch: Use same tokenizer for training and inference.

### Notebook Structure

1. **Introduction and Setup (Cells 1-10)**: Markdown, imports, device config.
2. **Data Exploration and Preparation (Cells 11-30)**: Loading datasets, preprocessing, tokenization, statistics.
3. **Model and Tokenizer Setup (Cells 31-50)**: Loading pre-trained models, configuration for fine-tuning/PEFT.
4. **Training and Fine-Tuning (Cells 51-100)**: Training loops, LoRA config, training args, checkpoint saving.
5. **Evaluation and Metrics (Cells 101-130)**: Inference, accuracy/F1, confusion matrices, reports.
6. **Additional Experiments and Visualizations (Cells 131-149)**: Hyperparameter sweeps, visualizations.

### Suggested Experiments

- LoRA sweeps: `r` in {4,8,16}, `alpha` in {8,16,32}.
- Class imbalance: Test class-weighted losses or oversampling.
- Temperature scaling on validation probabilities.

### Parts:

1. **Introduction and Setup** (Cells 1-10): Imports, device config, helpers.
2. **Data Exploration** (Cells 11-30): Loading CSV/HF datasets, preprocessing.
3. **Model & Tokenizer** (Cells 31-50): Loading pre-trained models.
4. **Training** (Cells 51-100): Fine-tuning with LoRA, training args.
5. **Evaluation** (Cells 101-130): Inference, metrics, confusion matrix.
6. **Experiments** (Cells 131-149): Visualizations, hyperparameter tests.

### Setup:

- Similar to CA1; supports CSV or HF datasets.
- Evaluation saves `misclassified_samples.csv`.

## CA3: LLM Homework 3

**Notebooks:**

- `Files/CAs/CA3/base_code/LLM_HW3_Part1_LLM_as_a_Judge.ipynb` (Base code for Part 1)
- `Files/CAs/CA3/base_code/LLM_HW3_Part2_RAG.ipynb` (Base code for Part 2)
- `Files/CAs/CA3/doCodes/LLM_HW3_Part1_LLM_as_a_Judge.ipynb` (Completed code for Part 1)
- `Files/CAs/CA3/doCodes/LLM_HW3_Part2_RAG.ipynb` (Completed code for Part 2)

This homework assignment focuses on advanced LLM applications: using LLMs as judges for evaluation and implementing Retrieval-Augmented Generation (RAG) systems.

### Project Structure

```
CA3/
├── base_code/
│   ├── LLM_HW3_Part1_LLM_as_a_Judge.ipynb
│   └── LLM_HW3_Part2_RAG.ipynb
├── doCodes/
│   ├── LLM_HW3_Part1_LLM_as_a_Judge.ipynb
│   └── LLM_HW3_Part2_RAG.ipynb
└── papers/
    ├── 2024.scichat-1.5.pdf
    ├── 2401.12178v1.pdf
    ├── 2401.15884v3.pdf
    └── 2401.18059v1.pdf
```

### Part 1: LLM as a Judge

Explores using LLMs for evaluation and judging tasks, such as scoring responses, assessing quality, and providing feedback on generated content.

#### Key Components:

- **Setup and Imports**: Environment configuration, library imports for LLM evaluation.
- **Model Loading**: Loading pre-trained models suitable for judging tasks.
- **Evaluation Pipelines**: Implementing scoring mechanisms, criteria definition, and automated assessment.
- **Results Analysis**: Analyzing judgment consistency, bias detection, and performance metrics.

#### Learning Objectives:

- Understand LLM capabilities in meta-evaluation tasks.
- Implement automated scoring systems.
- Analyze the reliability and limitations of LLM-based judging.

### Part 2: RAG (Retrieval-Augmented Generation)

Implements RAG systems that combine retrieval from knowledge bases with generative models for enhanced, factually grounded responses.

#### Key Components:

- **Data Indexing and Retrieval Setup**: Building vector databases, document chunking, and retrieval mechanisms.
- **Query Processing**: Handling user queries, relevance ranking, and context retrieval.
- **Generation with Retrieved Context**: Integrating retrieved information into prompt engineering for accurate generation.
- **Evaluation of RAG Performance**: Measuring retrieval accuracy, generation quality, and end-to-end system performance.

#### Learning Objectives:

- Master retrieval-augmented techniques to reduce hallucinations.
- Implement vector search and similarity matching.
- Optimize RAG pipelines for different domains and query types.

### Setup:

- Requires `langchain`, `faiss-cpu`, or similar libraries for RAG implementation.
- Additional libraries for vector stores, embeddings, and document processing.
- Python 3.9+, virtual environment recommended.

## CA4: Quantization and Self-Explanations

**Part 1 Notebook:** `Files/CAs/CA4/Uploaded Files/CA4_Part1_Quantization_Self_Explanations.ipynb`
**Part 2 Project:** `Files/CAs/CA4/Uploaded Files/CA4_Part2/` (Text-to-SQL with self-explanations)

This assignment covers model quantization techniques to reduce memory footprint and computational requirements, as well as self-explanation methods where LLMs explain their reasoning and decisions.

### Project Structure

```
CA4/
├── base_files/
│   ├── CA4_Part1_Quantization_Self_Explanations.ipynb
│   ├── CA4_Part2.zip
│   └── CA4_Part2 2/
│       ├── bird_loader.py
│       ├── db_manager.py
│       ├── evaluation.py
│       ├── method_run.py
│       ├── requirements.txt
│       ├── text2sql.ipynb
│       ├── __pycache__/
│       ├── data/
│       ├── evaluation/
│       └── nano-data/
├── Uploaded Files/
│   ├── CA4_Part1_Quantization_Self_Explanations.ipynb
│   └── CA4_Part2/
│       ├── .env
│       ├── add_analyser.py
│       ├── bird_loader.py
│       ├── db_manager.py
│       ├── evaluation.py
│       ├── method_run.py
│       ├── prompts.txt
│       ├── requirements.txt
│       ├── text2sql.ipynb
│       ├── ca4_env/
│       ├── data/
│       ├── evaluation/
│       └── nano-data/
└── [no README.md in structure]
```

### Part 1: Quantization and Self-Explanations

Focuses on model quantization techniques (8-bit, 4-bit) and implementing self-explanation capabilities in LLMs.

#### Key Components:

- **Quantization Setup**: Implementing various quantization levels using libraries like `bitsandbytes`.
- **Self-Explanation Methods**: Developing prompts and techniques for LLMs to explain their outputs and decision-making processes.
- **Evaluation of Quantized Models**: Comparing performance, accuracy, and efficiency of quantized vs. full-precision models.
- **Comparisons and Analysis**: Trade-offs between model size, speed, and quality.

#### Learning Objectives:

- Understand quantization techniques for model compression.
- Implement self-explanation mechanisms in LLMs.
- Analyze the impact of quantization on model performance.

### Part 2: Text-to-SQL with Self-Explanations

A comprehensive project implementing text-to-SQL conversion with self-explanatory capabilities, using the BIRD dataset for evaluation.

#### Project Components:

- **bird_loader.py**: Data loading utilities for the BIRD benchmark dataset.
- **db_manager.py**: Database management and SQL execution handling.
- **evaluation.py**: Evaluation scripts for SQL generation accuracy and self-explanations.
- **method_run.py**: Main execution script for running the text-to-SQL pipeline.
- **text2sql.ipynb**: Jupyter notebook demonstrating the text-to-SQL conversion process.
- **add_analyser.py**: Additional analysis tools.
- **prompts.txt**: Prompt templates for self-explanations.

#### Data Structure:

- **data/**: Full dataset with SQL databases and gold standard queries.
- **nano-data/**: Smaller subset for testing and development.
- **evaluation/**: Evaluation scripts and metrics.

#### Key Features:

- Text-to-SQL conversion using LLMs.
- Self-explanation generation for SQL queries.
- Evaluation on the BIRD benchmark.
- Support for multiple database schemas.

#### Learning Objectives:

- Implement advanced NLP tasks like text-to-SQL.
- Develop self-explanation capabilities in complex reasoning tasks.
- Evaluate LLM performance on structured prediction tasks.

### Setup:

- **Part 1**: Use `bitsandbytes` for quantization, ensure GPU support.
- **Part 2**: Python 3.9+, install from `requirements.txt` (includes transformers, torch, sqlite3, etc.).
- Virtual environment recommended for both parts.

## Papers

Each CA folder includes relevant research papers for deeper understanding of the topics covered. Below is a list of all papers included:

### CA1 Papers

- `Files/CAs/CA1/papers/2021.acl-long.353.pdf` - Research on tokenization and generation techniques.
- `Files/CAs/CA1/papers/2021.emnlp-main.243.pdf` - Paper on chat templates and instruction tuning.
- `Files/CAs/CA1/papers/2103.10385v2.pdf` - LoRA and PEFT methods for fine-tuning.

### CA2 Papers

- `Files/CAs/CA2/papers/2024.naacl-long.228.pdf` - Recent advances in LLM fine-tuning.
- `Files/CAs/CA2/papers/2201.11903v6.pdf` - Parameter-efficient fine-tuning techniques.

### CA3 Papers

- `Files/CAs/CA3/papers/2024.scichat-1.5.pdf` - LLM as a judge evaluation methods.
- `Files/CAs/CA3/papers/2401.12178v1.pdf` - Retrieval-augmented generation techniques.
- `Files/CAs/CA3/papers/2401.15884v3.pdf` - Advanced RAG implementations.
- `Files/CAs/CA3/papers/2401.18059v1.pdf` - Evaluation of RAG systems.

### CA4 Papers

- No papers folder in CA4 structure.

These papers provide theoretical background and research context for the practical implementations in each assignment.

## Setup and Requirements

### General Requirements

- **Python Version:** 3.9+ (some assignments may work with 3.8, but 3.9+ recommended)
- **Virtual Environment:** Strongly recommended to avoid dependency conflicts. Use `venv` or `conda`.
- **Hardware:** GPU recommended for training/fine-tuning tasks; CPU possible but slower.
- **Storage:** Sufficient space for model downloads (several GB) and datasets.

### Environment Setup

1. **Create Virtual Environment:**

   ```bash
   # Using venv
   python -m venv llm_course_env
   source llm_course_env/bin/activate  # Linux/Mac
   # llm_course_env\Scripts\activate   # Windows

   # Or using conda
   conda create -n llm_course python=3.11
   conda activate llm_course
   ```
2. **Upgrade pip:**

   ```bash
   pip install --upgrade pip
   ```

### Key Libraries Installation

Install core libraries used across assignments:

```bash
pip install torch torchvision torchaudio transformers datasets scikit-learn pandas matplotlib seaborn peft accelerate bitsandbytes
```

#### Assignment-Specific Requirements

**CA1 (Tokenization, Fine-Tuning):**

```bash
pip install torch transformers datasets peft accelerate scikit-learn pandas matplotlib seaborn
# For Hugging Face authentication
pip install huggingface_hub
```

**CA2 (Classification with LLMs):**

```bash
pip install torch transformers datasets scikit-learn pandas seaborn matplotlib accelerate peft
```

**CA3 (LLM as Judge, RAG):**

```bash
pip install langchain faiss-cpu sentence-transformers chromadb
# Additional RAG libraries
pip install unstructured pdfminer.six
```

**CA4 (Quantization, Text-to-SQL):**

```bash
pip install bitsandbytes torch transformers datasets sqlite3
# For CA4 Part 2
pip install python-dotenv
```

### GPU Setup

For CUDA support (if you have NVIDIA GPU):

```bash
# Install PyTorch with CUDA (check compatibility at pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For Apple Silicon (M1/M2 Macs):

```bash
# PyTorch with MPS support
pip install torch torchvision torchaudio
```

### Additional Setup Notes

- **Hugging Face Access:** Some models require authentication. Sign up at huggingface.co and obtain access tokens for gated models.
- **Model Downloads:** First run may download large models; ensure stable internet connection.
- **Memory Management:** For low-memory systems, use quantization or reduce batch sizes.
- **Jupyter Setup:** Install Jupyter if not already available:
  ```bash
  pip install jupyterlab
  ```

### Troubleshooting Installation

- **Import Errors:** Ensure virtual environment is activated and all packages installed.
- **CUDA Issues:** Verify CUDA version compatibility with PyTorch.
- **Memory Errors:** Reduce model size or use CPU-only versions.
- **Permission Errors:** Use `pip install --user` if system-wide installation fails.

## General Instructions

### Running the Notebooks

- **Environment:** Run notebooks in JupyterLab, Jupyter Notebook, or VS Code with Python extension.
- **Execution Order:** Execute cells sequentially; many cells depend on prior outputs and variable definitions.
- **Kernel Management:** Use a fresh kernel for each assignment to avoid variable conflicts.
- **Cell Execution:** Pay attention to markdown cells for explanations; code cells may take time for model loading/training.

### Evaluation and Outputs

- **Generated Files:** Check for output files like `misclassified_samples.csv`, confusion matrices, and plots.
- **Metrics Interpretation:** Understand accuracy, F1-score, precision, recall; analyze per-class performance.
- **Visualization:** Review plots for insights into model behavior, training progress, and comparative analysis.

### Experimentation and Customization

- **Hyperparameter Tuning:** Try different LoRA ranks (r=4,8,16), learning rates, batch sizes.
- **Data Variations:** Experiment with data augmentation, class balancing, or different datasets.
- **Model Choices:** Compare different base models, quantization levels, or fine-tuning strategies.
- **Advanced Techniques:** Implement early stopping, learning rate scheduling, or custom evaluation metrics.

### Troubleshooting Common Issues

- **Import Errors:** Restart kernel after installing packages; ensure virtual environment is active.
- **Memory Issues:** Reduce batch size, max sequence length, or use gradient accumulation; try quantization.
- **CUDA Errors:** Verify GPU drivers and PyTorch CUDA compatibility; fallback to CPU if needed.
- **Model Loading:** Check internet connection for downloads; ensure Hugging Face authentication for gated models.
- **Training Instability:** Lower learning rate, enable gradient clipping, or use mixed precision.
- **Evaluation Failures:** Verify data formats match expectations; check for missing labels or corrupted files.

### Performance Optimization

- **GPU Utilization:** Monitor GPU memory with `nvidia-smi`; use `torch.cuda.empty_cache()` to free memory.
- **Batch Processing:** Adjust batch sizes based on available memory; use dynamic batching for variable lengths.
- **Caching:** Enable dataset caching to avoid reprocessing; save model checkpoints for resuming training.
- **Quantization:** Use 8-bit/4-bit quantization for memory efficiency without significant quality loss.

### Academic Integrity and Best Practices

- **Original Work:** Ensure all code and analysis are your own; cite sources for external code/libraries.
- **Documentation:** Comment code clearly; document experimental setups, results, and conclusions.
- **Reproducibility:** Save random seeds, hyperparameters, and environment details for reproducible results.
- **Ethical Use:** Be aware of model biases; consider fairness and safety implications of LLM applications.

### Learning Tips

- **Incremental Learning:** Start with base code, understand each component before modifications.
- **Debugging:** Use print statements, logging, and visualization to understand model behavior.
- **Community Resources:** Refer to Hugging Face documentation, PyTorch tutorials, and research papers.
- **Version Control:** Use git to track changes; commit meaningful checkpoints.

For detailed per-assignment instructions, see individual READMEs in each CA folder.
