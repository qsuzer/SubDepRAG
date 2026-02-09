# SubDepRAG: Sub-question Dependency-Aware Retrieval-Augmented Generation for Multi-hop Question Answering

This repository contains the reproduction code for the SubDepRAG project and various baseline RAG pipelines. It implements a graph-based reasoning pipeline for Retrieval-Augmented Generation (RAG), allowing for dynamic decomposition, dependency analysis, and pruning of sub-questions.

![Model Framework](framework.pdf)

## Features

*   **Graph-Based Reasoning**: Models the reasoning process as a directed graph of sub-questions.
*   **Dynamic Pruning**: Automatically removes irrelevant sub-questions to reduce noise.
*   **Multiple Baselines**: Includes implementations of standard RAG, IRCoT, GenGround, PERQA, and DualRAG for comparison.
*   **Integrated Pipeline**: Built on top of `FlashRAG` for efficient inference.

## Installation

1.  Clone this repository.
2.  Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: You need to have `FlashRAG` installed.
- [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)

## Project Structure

```
GSub-RAG-Reproduce/
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── base_pipeline.py      # Base classes and shared logic
│   │   ├── subgraph_pipeline.py  # GSub-RAG (Graph-Based) implementation
│   │   ├── ircot_pipeline.py     # IRCoT implementation
│   │   ├── genground_pipeline.py # GenGround implementation
│   │   ├── perqa_pipeline.py     # PERQA implementation
│   │   ├── dualrag_pipeline.py   # DualRAG implementation
│   │   ├── rag_pipeline.py       # Standard RAG
│   │   ├── direct_pipeline.py    # Direct LLM (Zero-shot)
│   │   └── prompts.py            # Shared prompts
│   └── data/
│       └── ...
├── config/
│   └── basic_config.yaml         # Template configuration file
├── run_pipeline.py               # Main entry point for running pipelines
└── requirements.txt
```

## Supported Pipelines

This repository supports the following pipelines:

*   **`subgraph`**: The main GSub-RAG pipeline with graph-based reasoning.
*   **`ircot`**: Interleaving Retrieval with Chain-of-Thought (Iterative retrieval).
*   **`genground`**: Generate-and-Ground (Iterative generation with verification).
*   **`perqa`**: Planner-Executor-Reasoner architecture.
*   **`dualrag`**: Dual-view RAG (Context-aware and Answer-aware).
*   **`rag`**: Standard Retrieve-then-Generate.
*   **`direct`**: Direct LLM generation (Zero-shot).

## Usage

You can run any of the supported pipelines using the `run_pipeline.py` script.

### Basic Usage

```bash
python run_pipeline.py --pipeline <pipeline_name> --dataset <dataset_name>
```

### Examples

**Run the GSub-RAG (Subgraph) pipeline:**
```bash
python run_pipeline.py --pipeline subgraph --config config/basic_config.yaml
```

**Run the IRCoT pipeline on HotpotQA:**
```bash
python run_pipeline.py --pipeline ircot --dataset hotpotqa --split dev
```

**Run Standard RAG:**
```bash
python run_pipeline.py --pipeline rag --dataset hotpotqa
```

### Configuration

The configuration is managed via YAML files (e.g., `config/basic_config.yaml`). You can override specific settings via command-line arguments:

*   `--dataset`: Override the dataset name.
*   `--split`: Override the dataset split (e.g., `dev`, `test`).
*   `--gpu_id`: Specify GPU IDs.
*   `--test_sample_num`: Limit the number of samples for testing.

## Data Generation (Optional)

If you want to fine-tune your own model to perform decomposition and dependency analysis, you can use the scripts in `src/data/` to generate training data using a strong teacher model (e.g., GPT-4).

The seed data for fine-tuning is derived from the **PER-PSE** dataset: [https://huggingface.co/datasets/GenIRAG/PER-PSE](https://huggingface.co/datasets/GenIRAG/PER-PSE)
