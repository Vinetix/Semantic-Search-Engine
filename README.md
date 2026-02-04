# 🔍 Semantic Search Engine

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

An end-to-end **Semantic Search Engine** built on the **MS MARCO** dataset. This project demonstrates the evolution of a search system from a simple baseline to a production-grade architecture using **Contrastive Fine-Tuning**, **Hybrid Search (BM25 + Dense)**, and **Cross-Encoder Re-ranking**.

---

## 🚀 Key Features

* **Dense Retrieval:** Implemented a Bi-Encoder (`all-MiniLM-L6-v2`) architecture for semantic vector search.
* **Efficient Indexing:** Utilized **FAISS** (Facebook AI Similarity Search) for sub-millisecond vector lookups.
* **Contrastive Fine-Tuning:** Trained the model using **Hard Negative Mining** and `MultipleNegativesRankingLoss` to improve distinction between relevant and non-relevant documents.
* **Hybrid Search (Bonus):** Combined **BM25** (Sparse) and **Dense Embeddings** to capture both exact keyword matches and semantic meaning.
* **Two-Stage Re-Ranking (Bonus):** Implemented a Cross-Encoder re-ranker to maximize precision on top retrieval candidates.

---

## 📊 Performance Results

We evaluated the system on a held-out subset of MS MARCO queries using **MRR@10** (Mean Reciprocal Rank) and **NDCG@10**.

| Approach | MRR@10 | NDCG@10 | Latency (ms) | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.484 | 0.585 | ~20ms | Good speed, but misses specific nuances. |
| **Fine-Tuned** | 0.527 | 0.618 | ~17ms | **~9% Improvement** via domain adaptation. |
| **Hybrid (BM25)**| 0.465 | 0.573 | ~35ms | Best for queries requiring exact ID/keyword matches. |
| **Re-Ranker** | **0.560** | **0.654** | ~45ms | **Highest Accuracy** (+15% vs Baseline). |

> **Trade-off Analysis:** While the Re-ranker achieves the highest quality scores, it introduces higher latency. The Fine-Tuned Bi-Encoder offers the best balance of speed and accuracy for real-time applications.

---

## 🛠️ Technical Architecture

### 1. Data Ingestion
* **Dataset:** MS MARCO (Microsoft Machine Reading Comprehension).
* **Method:** Streamed via Hugging Face `datasets` to handle large-scale data (8.8M docs) without memory overflow.

### 2. Modeling Pipeline
* **Baseline:** Pre-trained `all-MiniLM-L6-v2`.
* **Training:** * **Loss Function:** `MultipleNegativesRankingLoss`.
    * **Strategy:** Mining hard negatives (top retrieved non-relevant docs) to force the model to learn subtle semantic distinctions.
* **Inference:**
    * **Stage 1:** Retrieve top 50 candidates via FAISS (Bi-Encoder).
    * **Stage 2:** Re-score top 50 pairs using a Cross-Encoder.

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* GPU recommended (Google Colab T4 is sufficient)

### Install dependencies
pip install torch pandas numpy sentence-transformers faiss-cpu rank_bm25 datasets

### Run the Notebook
The entire pipeline is contained within semantic_search_engine.ipynb. You upload it to Google Colab & run it.

### Clone the Repository
```bash
git clone https://github.com/Vinetix/Semantic-Search-Engine.git
