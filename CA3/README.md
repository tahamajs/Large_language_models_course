# CA3: LLM Homework 3

## Overview

This folder contains the third homework assignment for the Large Language Models (LLM) course. The assignment consists of two main parts: using LLMs as judges for evaluation tasks and implementing Retrieval-Augmented Generation (RAG) systems. This homework explores advanced applications of LLMs in meta-evaluation and knowledge-augmented generation.

## Table of Contents

- [Assignment Description](#assignment-description)
- [Project Structure](#project-structure)
- [Part 1: LLM as a Judge](#part-1-llm-as-a-judge)
- [Part 2: RAG (Retrieval-Augmented Generation)](#part-2-rag-retrieval-augmented-generation)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Papers](#papers)
- [Submission](#submission)

## Assignment Description

This homework assignment focuses on two advanced LLM applications:

1. **LLM as a Judge**: Using large language models to evaluate and score the outputs of other models or systems
2. **RAG Systems**: Implementing retrieval-augmented generation to enhance LLM responses with external knowledge

## Project Structure

```
CA3/
├── base_code/
│   ├── LLM_HW3_Part1_LLM_as_a_Judge.ipynb     # Base code for Part 1
│   └── LLM_HW3_Part2_RAG.ipynb                # Base code for Part 2
├── doCodes/
│   ├── LLM_HW3_Part1_LLM_as_a_Judge.ipynb     # Completed implementation for Part 1
│   └── LLM_HW3_Part2_RAG.ipynb                # Completed implementation for Part 2
├── papers/
│   ├── 2024.scichat-1.5.pdf                   # LLM evaluation methods
│   ├── 2401.12178v1.pdf                      # RAG techniques
│   ├── 2401.15884v3.pdf                      # Advanced RAG implementations
│   └── 2401.18059v1.pdf                      # RAG evaluation
└── README.md                                  # This file
```

## Part 1: LLM as a Judge

### Objective

Explore the use of LLMs for automated evaluation and judging tasks, including scoring model outputs, assessing response quality, and providing constructive feedback.

### Key Components

#### Setup and Environment

- Configure the development environment
- Install required libraries for LLM evaluation
- Set up model access and authentication

#### Model Loading and Configuration

- Load pre-trained models suitable for judging tasks
- Configure model parameters for evaluation scenarios
- Implement prompt engineering for consistent judging

#### Evaluation Pipeline Implementation

- Design scoring mechanisms and evaluation criteria
- Implement automated assessment workflows
- Create feedback generation systems

#### Results Analysis and Validation

- Analyze judgment consistency across different models
- Detect and mitigate potential biases in LLM judgments
- Validate evaluation reliability and performance metrics

### Learning Outcomes

- Understand the capabilities and limitations of LLMs in meta-evaluation tasks
- Implement automated scoring and feedback systems
- Analyze the reliability and consistency of LLM-based judgments

## Part 2: RAG (Retrieval-Augmented Generation)

### Objective

Implement and evaluate Retrieval-Augmented Generation systems that combine information retrieval with generative models to produce more accurate and factually grounded responses.

### Key Components

#### Data Preparation and Indexing

- Set up document collections and knowledge bases
- Implement document chunking and preprocessing
- Create vector embeddings for efficient retrieval

#### Retrieval System Implementation

- Build vector search and similarity matching capabilities
- Implement relevance ranking and filtering
- Optimize retrieval performance and accuracy

#### Generation with Context Integration

- Combine retrieved information with generative prompts
- Implement context-aware generation strategies
- Handle multi-document synthesis and summarization

#### System Evaluation and Optimization

- Measure retrieval accuracy and relevance
- Evaluate end-to-end generation quality
- Optimize system performance across different domains

### Learning Outcomes

- Master retrieval-augmented techniques to reduce hallucinations
- Implement efficient vector search and similarity matching
- Optimize RAG pipelines for various applications and domains

## Requirements

### System Requirements

- Python 3.9 or higher
- Minimum 8GB RAM (16GB recommended)
- GPU recommended for faster processing
- Stable internet connection for model downloads

### Software Dependencies

- PyTorch (with CUDA support for GPU)
- Transformers library
- LangChain or similar framework
- FAISS or ChromaDB for vector search
- Sentence transformers for embeddings
- Jupyter Notebook/Lab for development

## Installation

1. **Create Virtual Environment:**

   ```bash
   python -m venv ca3_env
   source ca3_env/bin/activate  # On Windows: ca3_env\Scripts\activate
   ```

2. **Install Core Dependencies:**

   ```bash
   pip install torch transformers datasets
   pip install langchain faiss-cpu sentence-transformers chromadb
   pip install jupyterlab matplotlib seaborn scikit-learn
   ```

3. **Install Additional Libraries:**

   ```bash
   pip install unstructured pdfminer.six
   pip install accelerate peft  # For model optimization
   ```

4. **Set up Jupyter:**
   ```bash
   pip install jupyterlab
   jupyter lab --generate-config
   ```

## Usage

### Running the Notebooks

1. **Launch Jupyter Lab:**

   ```bash
   jupyter lab
   ```

2. **Open Notebooks:**

   - For Part 1: `doCodes/LLM_HW3_Part1_LLM_as_a_Judge.ipynb`
   - For Part 2: `doCodes/LLM_HW3_Part2_RAG.ipynb`

3. **Execution Order:**
   - Start with setup and import cells
   - Run data loading and preprocessing
   - Execute model loading and configuration
   - Run evaluation or generation pipelines
   - Analyze results and visualizations

### Base Code vs. Completed Code

- **base_code/**: Contains starter templates and basic implementations
- **doCodes/**: Contains completed, fully functional implementations

Use base_code for understanding the structure, and doCodes for running complete examples.

## Evaluation

### Part 1 Evaluation Criteria

- Correctness of judging mechanisms
- Consistency of evaluation scores
- Quality of generated feedback
- Analysis of judgment biases and limitations

### Part 2 Evaluation Criteria

- Retrieval accuracy and relevance
- Generation quality with retrieved context
- System performance and efficiency
- Comparative analysis with baseline methods

### Metrics and Outputs

- Automated evaluation scores
- Human evaluation comparisons
- Performance benchmarks
- Qualitative analysis of generated outputs

## Papers

The following research papers provide theoretical background:

1. **2024.scichat-1.5.pdf**: Methods for using LLMs in scientific evaluation and peer review
2. **2401.12178v1.pdf**: Recent advances in retrieval-augmented generation techniques
3. **2401.15884v3.pdf**: Advanced implementations of RAG systems with improved retrieval
4. **2401.18059v1.pdf**: Comprehensive evaluation frameworks for RAG systems

## Submission

### Deliverables

- Completed Jupyter notebooks with all implementations
- Detailed analysis and results documentation
- Performance evaluation reports
- Code documentation and comments

### Grading Criteria

- Technical correctness (40%)
- Implementation quality (30%)
- Analysis and insights (20%)
- Documentation and presentation (10%)

## Troubleshooting

### Common Issues

- **Memory Errors**: Reduce batch sizes or use smaller models
- **Import Errors**: Ensure all dependencies are installed in the virtual environment
- **Model Loading**: Check internet connection and Hugging Face authentication
- **Performance Issues**: Use GPU acceleration or optimize code for CPU usage

### Getting Help

- Check the main course README for general setup instructions
- Review the base_code notebooks for implementation examples
- Consult the research papers for theoretical understanding
- Contact course TAs for technical assistance

---

**Student:** Mohammad Taha Majlesi (810101504)  
**Assignment:** CA3 - LLM Homework 3  
**Date:** September 2025

Good luck with CA3! This assignment will deepen your understanding of advanced LLM applications.
