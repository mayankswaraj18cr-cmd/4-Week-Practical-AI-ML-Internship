# SkillSetGo AI/ML Internship

An implementation-first, four-week AI/ML curriculum covering classical machine learning, deep learning, NLP, transformers, retrieval-augmented generation, and agent systems. The repository is a small, testable monorepo designed for GitHub Codespaces and local Python development.

## What This Repository Provides

- Typed, defensive Python examples that can be reused as library code.
- A reproducible development container with Python 3.11 and VS Code tooling.
- Pinned dependencies for NumPy, pandas, scikit-learn, PyTorch, Transformers, vector databases, and testing.
- A week-by-week pytest suite that runs without network access or API credentials.
- Lazy loading for heavyweight Hugging Face model downloads.
- A dependency-light vector store interface that can be replaced by FAISS or Chroma.

## Architecture

```text
skillsetgo-aiml-internship/
├── .devcontainer/                  # Reproducible Python 3.11 Codespace
├── week_1_foundations/             # Python, data, math, classical ML
│   ├── python_fundamentals.py
│   ├── data_cleaning_eda.py
│   ├── math_for_ml.py
│   └── scikit_learn_models.py
├── week_2_deep_learning/           # PyTorch, CNNs, experiment logging
├── week_3_nlp_transformers/        # NLP, attention, Hugging Face
├── week_4_llms_rag_agents/         # Retrieval, RAG, tool-using agents
├── tests/                          # Week-by-week pytest verification
├── requirements.txt
└── README.md
```

## Quick Start: GitHub Codespaces

1. Push or open this repository on GitHub.
2. Select **Code > Create codespace on main**.
3. Wait for the `.devcontainer` build to finish. It installs Python 3.11, Pylance, Python, and Jupyter extensions, then installs `requirements.txt`.
4. Run the verification suite:

```bash
python --version
pytest -q tests/
```

### Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q tests/
```

PyTorch tests are skipped automatically when PyTorch is unavailable. Install the full requirements inside Codespaces to execute them.

## Four-Week Milestones

| Week | Milestone | Deliverables |
| --- | --- | --- |
| 1 | Build reliable ML foundations | Typed Python records, pandas cleaning, NumPy calculus, Ridge and Random Forest pipelines |
| 2 | Train neural networks | Batch-normalized MLP, CNN feature extractor, optimizer experiment logger |
| 3 | Understand modern NLP | Normalization, TF-IDF, embeddings, self-attention, Hugging Face Trainer factory |
| 4 | Ship an LLM application core | Chunking, local vector retrieval, RAG prompt construction, bounded agent tool loop |

### Week 1: Foundations

Learn reliable Python, data preparation, linear algebra, and classical supervised learning. The week includes validated records, pandas imputation and IQR handling, NumPy matrix/eigenvalue/MSE helpers, and Ridge and Random Forest pipeline factories.

### Week 2: Deep Learning and Vision

Build a BatchNorm/Dropout feed-forward network, a CNN with reusable feature extraction, DataLoader training utilities, and JSON experiment logging for Adam, AdamW, and SGD.

### Week 3: NLP and Transformers

Normalize text, create TF-IDF and demonstration embeddings, implement multi-head scaled dot-product attention, and construct Hugging Face sequence-classification trainers with lazy model loading.

### Week 4: RAG and Agents

Chunk documents, retrieve normalized local embeddings, inject evidence into a constrained RAG prompt, and execute validated tool calls with a hard maximum step limit.

## Running Modules

Each module is importable and can also be inspected directly:

```bash
python -m week_1_foundations.python_fundamentals
python -m week_2_deep_learning.pytorch_neural_network
python -m week_3_nlp_transformers.transformer_attention_from_scratch
python -m week_4_llms_rag_agents.rag_pipeline
pytest tests/
```

Run a single week:

```bash
pytest -q tests/test_week_1.py
pytest -q tests/test_week_2.py
pytest -q tests/test_week_3.py
pytest -q tests/test_week_4.py
```

Use the modules from Python:

```python
from week_1_foundations.data_cleaning_eda import impute_missing_values
from week_3_nlp_transformers.text_preprocessing_embeddings import tfidf_embeddings
from week_4_llms_rag_agents.vector_db_retrieval import LocalVectorStore, hashing_embeddings
```

The Hugging Face and model-training helpers download models only when explicitly called. Week 4 uses a deterministic local vector baseline so its tests run without API keys; the `LocalVectorStore` interface can be replaced with FAISS or Chroma in deployment.

## Engineering Notes

- Public functions validate inputs and use type annotations.
- Optional heavyweight model loading is lazy to keep imports and unit tests fast.
- Training artifacts, caches, logs, checkpoints, and virtual environments are excluded by `.gitignore`.
- The included tests verify core behavior without network access or external credentials.

## Production Extensions

The curriculum keeps infrastructure intentionally small. A production service should add model and data versioning, structured logging, metrics, access controls, secret management, persistent vector storage, evaluation datasets, prompt/version tracking, and CI execution of the test suite.