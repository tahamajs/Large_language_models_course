# CA1: Large Language Models Exploration

## Overview

This repository contains the final submission for CA1 in the Large Language Models (LLM) course. The assignment explores various aspects of Large Language Models, including tokenization, model generation, fine-tuning techniques, and performance evaluation. The project demonstrates hands-on experience with modern LLM techniques using the Llama-3.2-1B models and the emotion detection dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Experiments](#key-experiments)
- [Results and Findings](#results-and-findings)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running this notebook, ensure you have the following:

- Python 3.8 or higher
- Access to Hugging Face models (specifically `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-1B-Instruct`, `mistralai/Mistral-7B-v0.1`, `microsoft/Phi-4-mini-instruct`)
- Sufficient GPU memory (recommended: at least 8GB VRAM for full fine-tuning, less for LoRA)
- Hugging Face account with model access tokens

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/tahamajs/Large_language_models_course.git
   cd Large_language_models_course/Files/CAs/CA1/code/
   ```

2. Create a virtual environment:

   ```bash
   python -m venv ca1_env
   source ca1_env/bin/activate  # On Windows: ca1_env\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up Hugging Face authentication:
   - Obtain access tokens for the required models from [Hugging Face](https://huggingface.co/settings/tokens)
   - Set the tokens as environment variables or use the notebook's authentication cell

## Project Structure

```
CA1/
├── code/
│   ├── LLM_CA1_final.ipynb    # Main notebook with all experiments
│   ├── README.md              # This file
│   └── requirements.txt       # Python dependencies
└── [other directories...]
```

## Usage

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open `LLM_CA1_final.ipynb` in your browser

3. Follow the cells in order:
   - Start with setting up authentication and imports
   - Run tokenization experiments
   - Compare base vs. instruction-tuned models
   - Perform LoRA fine-tuning on emotion detection
   - Evaluate and compare model performances

**Note:** Some cells may take significant time to run, especially the fine-tuning sections. Ensure you have a stable internet connection for downloading models.

## Key Experiments

The notebook is structured around the CA1 assignment questions, covering comprehensive exploration of LLMs:

### Q0: Setting Up

- **Environment configuration and package installation**: Install required libraries including transformers, peft, datasets, and accelerate for model handling and fine-tuning
- **Hugging Face model access setup**: Guide users through obtaining and configuring access tokens for gated models like Llama-3.2 variants
- **Device detection (CPU/GPU/MPS)**: Automatic detection and utilization of available hardware acceleration for optimal performance

### Q1: First Steps (25 pts)

#### Q1.1: Readable Model Generation (1 pt)

- **Understanding model token outputs vs. human-readable text**: Learn how LLMs generate token IDs instead of readable text, requiring decoding for interpretation
- **Using tokenizer decode function**: Implement proper decoding techniques to convert model outputs to human-readable format
- **Handling input prompt separation in outputs**: Techniques to isolate generated content from the original input prompt in model responses

#### Q1.2: Generation Function (1 pt)

- **Creating a reusable text generation function**: Develop a modular function that can be used throughout the notebook for consistent text generation
- **Incorporating generation configurations**: Experiment with parameters like temperature, top-k, top-p, and max_length to control generation behavior
- **Testing with custom prompts**: Validate the generation function with various prompts to ensure reliability and flexibility

#### Q1.3: Comparing Different Tokenizers (3 pts)

- **Tokenization analysis across model families**: Examine how different LLM architectures handle text tokenization, including BPE, WordPiece, and SentencePiece methods
- **Persian text tokenization comparison**: Evaluate how multilingual tokenizers handle non-Latin scripts, particularly Persian/Farsi text with complex morphology
- **Token efficiency and readability assessment**: Compare token counts, subword splitting, and semantic preservation across different tokenization strategies

#### Q1.4: Base Model vs. Instruction-Tuned Model (10 pts)

- **Performance comparison between base and instruction-tuned Llama models**: Analyze differences in response quality, coherence, and task-following capabilities
- **Analysis of response styles and verbosity**: Study how instruction tuning affects response length, structure, and conciseness
- **Impact of temperature on output consistency**: Investigate how generation parameters interact differently with base vs. instruction-tuned models

#### Q1.5: Chat Templates for Instruction Models (10 pts)

- **Understanding chat templates and their role**: Learn about standardized formats for multi-turn conversations that help models maintain context and roles
- **Implementing ChatML format for conversations**: Apply proper message formatting with system, user, and assistant roles for structured interactions
- **Comparing template vs. non-template prompting**: Demonstrate the importance of proper formatting for consistent and high-quality model responses
- **Multi-turn conversation simulation with role-playing**: Create complex dialogues simulating expert debates to test model capabilities in maintaining context

### Q2: Fine-Tuning Using LoRA (75 pts)

#### Q2.0: Utilities (5 pts)

- **Stratified sampling implementation for dataset balancing**: Create functions to maintain class distribution when splitting datasets for training and evaluation
- **Maintaining class proportions in train/test splits**: Ensure representative samples across all emotion categories to prevent biased model training

#### Q2.1: Preparing Data for Fine-Tuning (10 pts)

- **Conversation formatting for instruction tuning**: Structure emotion detection as instruction-response pairs suitable for causal language modeling
- **Tokenization with masking for causal LM training**: Prepare input sequences with proper attention masks and label masking to focus training on generated content
- **Tokenizer parameter optimization**: Experiment with truncation, padding, and max_length settings for efficient batch processing and memory usage
- **Label masking rationale and implementation**: Understand why masking instruction tokens prevents the model from learning to copy prompts instead of generating responses

#### Q2.2: Experimenting with LoRA Configuration (3 pts)

- **Rank (r) parameter exploration**: Test different low-rank dimensions to find the optimal balance between model capacity and parameter efficiency
- **LoRA alpha scaling factor analysis**: Investigate how the scaling parameter affects update magnitude and training stability
- **Target module selection**: Compare fine-tuning different transformer components (attention layers vs. feed-forward networks)
- **Dropout effects on training stability**: Analyze how regularization in LoRA adapters impacts overfitting and generalization

#### Q2.3: Training Callbacks and Early Stopping (10 pts)

- **Custom callback implementation for perplexity tracking**: Build monitoring tools to track training progress and model performance in real-time
- **Early stopping based on validation loss**: Implement automatic training termination to prevent overfitting and save computational resources
- **Training monitoring and logging**: Set up comprehensive logging for loss, perplexity, and other metrics throughout the training process

#### Q2.4: Training Arguments (7 pts)

- **Hyperparameter optimization for LoRA fine-tuning**: Fine-tune learning rates, batch sizes, and other training parameters for optimal convergence
- **Learning rate scheduling strategies**: Compare different scheduling approaches (linear, cosine) for stable training progress
- **Memory-efficient training configurations**: Configure training to work within hardware constraints using gradient accumulation and mixed precision

#### Q2.5: Memory Usage Analysis (8 pts)

- **Full fine-tuning vs. LoRA memory requirements**: Quantify the dramatic reduction in GPU memory usage achieved through parameter-efficient fine-tuning
- **GPU memory estimation for different scenarios**: Calculate memory footprints for various model sizes and training configurations
- **Parameter efficiency quantification**: Demonstrate the percentage of trainable parameters in LoRA vs. full fine-tuning approaches

#### Q2.6: Training the Model (2 pts)

- **LoRA adapter training execution**: Run the complete fine-tuning pipeline on the emotion detection dataset
- **Model saving and checkpointing**: Implement proper model persistence for later inference and evaluation

#### Q2.7: IA3 Method (2 pts)

- **Explanation of Infused Adapter by Inhibiting and Amplifying Inner Activations**: Describe how IA3 uses learned vectors to rescale activations within transformer layers
- **Comparison with LoRA approach**: Contrast IA3's element-wise rescaling with LoRA's low-rank matrix adaptation strategy

#### Q2.8: Soft Prompt Methods (4 pts)

- **Prompt Tuning analysis**: Examine how learned continuous prompts can steer frozen models toward specific tasks
- **Prefix Tuning explanation**: Understand how prepending learned vectors to all transformer layers enables deeper model conditioning
- **P-Tuning overview**: Explore the combination of continuous prompts with discrete template structures
- **Key differences between soft prompt techniques**: Compare the trade-offs between different prompt-based adaptation methods

#### Q2.9: Generating Output from Models (10 pts)

- **Inference pipeline for emotion detection**: Build end-to-end evaluation system for comparing model performance on the emotion classification task
- **Output parsing and label extraction**: Develop robust methods to extract emotion labels from generated text responses
- **Comparative evaluation across model variants**: Systematically compare base, instruction-tuned, and fine-tuned model performance

#### Q2.10: Performance Comparison Visualization (4 pts)

- **Accuracy and F1-score metrics calculation**: Implement proper evaluation metrics for multi-class emotion classification
- **Grouped bar chart visualization**: Create clear visual comparisons of model performance across different metrics
- **Model performance analysis**: Interpret results and draw conclusions about the effectiveness of different fine-tuning approaches

### Additional Features

- **Academic integrity disclosure**: Transparent reporting of AI assistance and external help used in the assignment
- **Comprehensive markdown documentation**: Detailed explanations and insights throughout the notebook
- **Code comments and modular design**: Well-documented code with reusable functions and clear structure

## Results and Findings

### Key Insights:

1. **Tokenization**: Llama models show superior handling of Persian text compared to Mistral and Phi models.

2. **Model Behavior**: Instruction-tuned models provide more focused and concise responses, while base models tend to be more verbose.

3. **Fine-Tuning Efficiency**: LoRA enables effective fine-tuning with only ~0.9% of model parameters trainable, significantly reducing memory requirements.

4. **Performance Gains**: Fine-tuned models achieve substantial improvements over base models on emotion detection tasks.

### Performance Metrics:

- Base Model: ~XX% accuracy
- Instruction-Tuned Model: ~XX% accuracy
- LoRA Fine-Tuned Model: ~XX% accuracy

_(Exact metrics depend on training runs and hyperparameters)_

## Contributing

This is an academic assignment submission. For questions or clarifications, please contact the course TAs:

- vahyd@live.com
- amirh.bonakdar@ut.ac.ir

## License

This project is part of an academic course assignment. Please refer to the course policy regarding code sharing and academic integrity.

## Acknowledgments

- Course instructors and TAs for guidance
- Hugging Face for providing the models and datasets
- Meta, Mistral, and Microsoft for open-sourcing their language models
- The emotion dataset creators and maintainers

---

**Student:** Mohammad Taha Majlesi  
**Student ID:** 810101504  
**Course:** Large Language Models (LLM)  
**Assignment:** CA1  
**Date:** September 2025
