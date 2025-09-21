# CA4: Quantization and Self-Explanations

## Overview

This folder contains the fourth course assignment focusing on model quantization techniques and self-explanation methods for Large Language Models (LLMs). The assignment consists of two main parts: implementing model quantization for efficiency and developing self-explanatory capabilities in LLMs for better interpretability.

## Table of Contents

- [Assignment Description](#assignment-description)
- [Project Structure](#project-structure)
- [Part 1: Quantization and Self-Explanations](#part-1-quantization-and-self-explanations)
- [Part 2: Text-to-SQL with Self-Explanations](#part-2-text-to-sql-with-self-explanations)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Submission](#submission)

## Assignment Description

This assignment explores two critical aspects of modern LLM deployment and interpretability:

1. **Model Quantization**: Techniques to reduce model size and computational requirements while maintaining performance
2. **Self-Explanations**: Methods for LLMs to explain their reasoning and decision-making processes

## Project Structure

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
│       │   ├── data_gold.sql
│       │   ├── dataset.json
│       │   ├── tables.json
│       │   └── dev_databases/
│       ├── evaluation/
│       │   ├── evaluation_ves.py
│       │   └── evaluation.py
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
│       │   ├── data_gold.sql
│       │   ├── dataset.json
│       │   ├── predict_data.json
│       │   ├── tables.json
│       │   └── dev_databases/
│       ├── evaluation/
│       │   ├── evaluation_ves.py
│       │   └── evaluation.py
│       └── nano-data/
└── README.md
```

## Part 1: Quantization and Self-Explanations

### Objective

Implement model quantization techniques to reduce memory footprint and computational requirements, while developing methods for LLMs to explain their outputs and decision-making processes.

### Key Components

#### Quantization Implementation

- Set up 8-bit and 4-bit quantization using bitsandbytes
- Compare performance of quantized vs. full-precision models
- Analyze trade-offs between model size, speed, and accuracy

#### Self-Explanation Methods

- Develop prompting strategies for model explanations
- Implement chain-of-thought reasoning for transparency
- Create methods to extract and present model reasoning

#### Evaluation and Analysis

- Benchmark quantized model performance
- Assess explanation quality and usefulness
- Compare different quantization levels and explanation methods

### Learning Outcomes

- Understand quantization techniques for model compression
- Implement self-explanation mechanisms in LLMs
- Analyze the impact of quantization on model performance and efficiency

## Part 2: Text-to-SQL with Self-Explanations

### Objective

Build a comprehensive text-to-SQL conversion system with self-explanatory capabilities, evaluated on the BIRD benchmark dataset. The system should convert natural language queries to SQL statements while providing explanations of the conversion process.

### Project Components

#### Core Modules

**bird_loader.py**

- Data loading utilities for the BIRD benchmark dataset
- Handles dataset parsing and preprocessing
- Manages different database schemas and query types

**db_manager.py**

- Database connection and management
- SQL execution handling and result processing
- Schema information extraction and validation

**evaluation.py**

- Evaluation metrics for SQL generation accuracy
- Self-explanation quality assessment
- Performance benchmarking against gold standards

**method_run.py**

- Main execution script for the text-to-SQL pipeline
- Orchestrates the entire conversion and explanation process
- Handles batch processing and result aggregation

**text2sql.ipynb**

- Jupyter notebook demonstrating the complete pipeline
- Interactive examples and visualizations
- Step-by-step walkthrough of the conversion process

**add_analyser.py**

- Additional analysis and debugging tools
- Performance profiling and error analysis
- Result visualization and reporting

**prompts.txt**

- Prompt templates for text-to-SQL conversion
- Self-explanation prompt engineering
- Instruction templates for different query types

#### Data Structure

**data/**

- Full BIRD dataset with SQL databases
- Gold standard queries and expected results
- Database schema definitions and metadata

**nano-data/**

- Smaller subset for development and testing
- Representative samples from different domains
- Reduced computational requirements for experimentation

**evaluation/**

- Evaluation scripts and metrics
- Automated testing frameworks
- Result comparison and analysis tools

### Key Features

- **Text-to-SQL Conversion**: Natural language to SQL translation
- **Self-Explanations**: Generated explanations for SQL queries
- **Multi-Domain Support**: Handles various database schemas
- **Evaluation Framework**: Comprehensive benchmarking on BIRD dataset
- **Modular Architecture**: Separated concerns for maintainability

### Learning Outcomes

- Implement advanced NLP tasks like text-to-SQL conversion
- Develop self-explanation capabilities in complex reasoning tasks
- Evaluate LLM performance on structured prediction tasks
- Build modular, scalable ML systems

## Requirements

### System Requirements

- Python 3.9 or higher
- Minimum 16GB RAM (32GB recommended for full datasets)
- GPU strongly recommended for training and inference
- Sufficient storage for datasets and model downloads

### Software Dependencies

- PyTorch with CUDA support
- Transformers library
- BitsAndBytes for quantization
- SQLite for database operations
- Jupyter Notebook/Lab
- Required Python packages (see requirements.txt)

## Installation

1. **Create Virtual Environment:**

   ```bash
   python -m venv ca4_env
   source ca4_env/bin/activate  # On Windows: ca4_env\Scripts\activate
   ```

2. **Install Dependencies:**

   ```bash
   # For Part 1
   pip install torch transformers bitsandbytes accelerate

   # For Part 2
   pip install torch transformers datasets sqlite3 python-dotenv
   pip install jupyterlab pandas numpy matplotlib seaborn
   ```

3. **Install from requirements.txt (Part 2):**

   ```bash
   cd "Uploaded Files/CA4_Part2"
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   ```bash
   # Create .env file in CA4_Part2 directory
   echo "OPENAI_API_KEY=your_key_here" > .env
   # Add other required environment variables
   ```

## Usage

### Part 1: Quantization Notebook

1. **Open Notebook:**

   ```bash
   jupyter lab "Uploaded Files/CA4_Part1_Quantization_Self_Explanations.ipynb"
   ```

2. **Execution Steps:**
   - Run setup and import cells
   - Load and quantize models
   - Test self-explanation methods
   - Compare performance metrics

### Part 2: Text-to-SQL Project

1. **Navigate to Project Directory:**

   ```bash
   cd "Uploaded Files/CA4_Part2"
   ```

2. **Run Main Script:**

   ```bash
   python method_run.py
   ```

3. **Interactive Notebook:**

   ```bash
   jupyter lab text2sql.ipynb
   ```

4. **Evaluation:**
   ```bash
   python evaluation/evaluation.py
   ```

## Datasets

### BIRD Dataset (Part 2)

- **Full Dataset**: Located in `data/` directory
- **Development Set**: SQLite databases with gold standard SQL queries
- **Nano Dataset**: Smaller subset in `nano-data/` for quick testing
- **Features**:
  - Multiple domains (student club, superhero, toxicology)
  - Complex SQL queries with joins, aggregations, subqueries
  - Schema information and table descriptions

### Data Format

- **dataset.json**: Contains natural language questions and SQL pairs
- **tables.json**: Database schema information
- **data_gold.sql**: Gold standard SQL queries for evaluation

## Evaluation

### Part 1 Metrics

- Model size reduction (parameters, memory usage)
- Inference speed improvements
- Performance degradation analysis
- Explanation quality assessment

### Part 2 Metrics

- **SQL Accuracy**: Exact match, execution accuracy
- **Explanation Quality**: Clarity, correctness, completeness
- **BIRD Benchmark Scores**: Official evaluation metrics
- **Performance Metrics**: Latency, throughput, resource usage

### Evaluation Scripts

- Automated testing frameworks
- Result comparison with baselines
- Statistical significance testing
- Qualitative analysis tools

## Submission

### Part 1 Deliverables

- Completed quantization notebook
- Performance comparison reports
- Self-explanation method implementations
- Analysis of trade-offs and findings

### Part 2 Deliverables

- Complete text-to-SQL system
- Self-explanation implementations
- Evaluation results on BIRD dataset
- Documentation and code comments

### Grading Criteria

- Technical implementation (40%)
- Performance and accuracy (30%)
- Analysis and insights (20%)
- Documentation quality (10%)

## Troubleshooting

### Common Issues

- **Quantization Errors**: Ensure compatible PyTorch and CUDA versions
- **Memory Issues**: Use smaller batch sizes or gradient accumulation
- **Database Connection**: Verify SQLite installation and file paths
- **API Limits**: Check rate limits for external API calls

### Performance Optimization

- Use GPU acceleration for faster processing
- Implement batch processing for efficiency
- Cache embeddings and intermediate results
- Profile code to identify bottlenecks

### Getting Help

- Review the main course README for general guidance
- Check individual component documentation
- Consult the BIRD dataset paper for baseline methods
- Contact course instructors for technical support

---

**Student:** Mohammad Taha Majlesi (810101504)  
**Assignment:** CA4 - Quantization and Self-Explanations  
**Date:** September 2025

This assignment will challenge you to optimize LLMs for real-world deployment while making them more interpretable and trustworthy.
