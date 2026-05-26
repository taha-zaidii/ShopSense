# ShopSense — Neural Collaborative Filtering Recommender

> Personalized e-commerce product recommendations built on the **Neural Collaborative Filtering (NeuMF)** architecture from He et al. (2017). Fuses Generalized Matrix Factorization (linear latent factors) with a Multi-Layer Perceptron (non-linear interactions) under one trainable model — then evaluates it with proper ranking metrics, not just MSE.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

---

## Why NeuMF over plain matrix factorization

Classical matrix factorization captures **linear** user↔item interactions: the predicted preference is just an inner product of two latent vectors. That's elegant, but it can't model the higher-order, non-linear patterns that show up in real e-commerce data (e.g. *"users who bought A and C, but not B, tend to like D"*).

NeuMF fixes this by running **two pathways in parallel** and concatenating them before the final prediction layer:

```
                     user_id          item_id
                        │                │
              ┌─────────┴─────┐  ┌───────┴────────┐
              ↓               ↓  ↓                ↓
        ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ GMF user │   │ GMF item │   │ MLP user │   │ MLP item │
        │  embed   │   │  embed   │   │  embed   │   │  embed   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │              │
             └───── ⊙ ──────┘              └──── concat ──┘
                     │                              │
                     │                              ↓
                     │                       [ MLP layers ]
                     │                              │
                     └──────────── concat ──────────┘
                                       │
                                       ↓
                                 sigmoid → ŷ
```

- **GMF pathway** — element-wise product of user and item embeddings (the matrix-factorization signal)
- **MLP pathway** — concatenated embeddings through stacked dense layers (the non-linear signal)
- **Fusion layer** — both pathways concatenated and passed through a final sigmoid for prediction

---

## What's in the box

| Component | Purpose |
|---|---|
| **`src/`** | Data preprocessing, model architecture, training loop, evaluation |
| **`product-recommendation-system.ipynb`** | End-to-end notebook: EDA → training → evaluation → recommendations |
| **`app.py`** | Flask web app for interactive training + recommendation queries |
| **`templates/`, `static/`** | Web UI for non-notebook users |
| **`verify_recommendations.py`** | Sanity-check script for trained model output |

---

## Evaluation — done right

Recommenders are notoriously easy to fool with the wrong metric. ShopSense evaluates on **two metric families**, not just one:

| Family | Metrics | Why it matters |
|---|---|---|
| **Regression accuracy** | MAE · MSE · RMSE | How close are predicted ratings to actuals? |
| **Ranking quality** | Precision@K · Recall@K · NDCG | Are the *right* items at the *top* of the recommendation list? |

A model can have low RMSE but still produce useless recommendations if it ranks irrelevant items first. Tracking both prevents that failure mode.

---

## Quick Start

```bash
# Clone
git clone https://github.com/taha-zaidii/ShopSense.git
cd ShopSense

# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dataset — download Amazon Electronics CSV
# Columns required: User_id, Product_id, rating, Timestamp
# Place at: data/ratings.csv

# Option A — Jupyter notebook (EDA + training + analysis)
jupyter notebook product-recommendation-system.ipynb

# Option B — Flask web app (train + serve recommendations)
python app.py
# → http://127.0.0.1:5000
```

---

## Stack

| Layer | Tech |
|---|---|
| **Model** | TensorFlow 2.x · Keras |
| **Architecture** | NeuMF (GMF + MLP fusion) |
| **Data** | Amazon Electronics ratings dataset |
| **Serving** | Flask 3.x |
| **Analysis** | Jupyter · pandas · NumPy · scikit-learn |
| **Python** | 3.11+ |

---

## References

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). **Neural Collaborative Filtering.** *Proceedings of the 26th International Conference on World Wide Web (WWW '17).* https://arxiv.org/abs/1708.05031
- Koren, Y., Bell, R., & Volinsky, C. (2009). **Matrix Factorization Techniques for Recommender Systems.** *Computer, 42(8), 30-37.* (Netflix Prize era)

---

## Project Context

Final project for **Recommender Systems**, Fall 2025, FAST NUCES Karachi.

**Authors:** Syed Taha Zaidi · Shahmeer Irfan

## License

MIT — see [LICENSE](LICENSE).
