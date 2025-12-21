<p align="center">
  <h1 align="center">🛍️ ShopSense</h1>
  <p align="center"><strong>Personalized Product Recommendation Engine</strong></p>
  <p align="center">
    <em>Neural Collaborative Filtering with Matrix Factorization for E-Commerce</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-3.x-green?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 About

**ShopSense** is a personalized product recommendation system that uses **Neural Collaborative Filtering (NeuMF)** to predict user preferences and recommend products. Built as a final project for the **Recommender Systems** course (Fall 2025), it demonstrates key concepts including:

- **Matrix Factorization** for latent factor extraction
- **Neural Collaborative Filtering** for deep learning-based recommendations
- **Hybrid approach** combining MLP and GMF pathways
- **Comprehensive evaluation** using ranking metrics

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **NeuMF Model** | Combines GMF (Generalized Matrix Factorization) + MLP for accurate predictions |
| 📊 **Verification System** | Evaluate recommendations with Precision@K, Recall@K, NDCG, Hit Rate |
| 🌐 **Web Interface** | Beautiful, responsive UI for training and getting recommendations |
| ⚡ **Real-time Predictions** | Get personalized recommendations instantly |
| 📈 **Model Metrics** | Track MAE, MSE, RMSE during training |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ShopSense Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐     ┌──────────┐                             │
│   │  User ID │     │Product ID│                             │
│   └────┬─────┘     └────┬─────┘                             │
│        │                │                                    │
│   ┌────▼────────────────▼────┐                              │
│   │     Embedding Layers     │                              │
│   └────┬────────────────┬────┘                              │
│        │                │                                    │
│   ┌────▼────┐      ┌────▼────┐                              │
│   │   GMF   │      │   MLP   │                              │
│   │ (Dot)   │      │ (Dense) │                              │
│   └────┬────┘      └────┬────┘                              │
│        │                │                                    │
│   ┌────▼────────────────▼────┐                              │
│   │       Concatenate        │                              │
│   └───────────┬──────────────┘                              │
│               │                                              │
│   ┌───────────▼──────────────┐                              │
│   │    Predicted Rating      │                              │
│   │       (1-5 scale)        │                              │
│   └──────────────────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Concepts Used

This project implements key concepts from the Recommender Systems course:

### 1. **Collaborative Filtering**
- Uses user-item interaction data (ratings) to find patterns
- Addresses the cold-start problem through learned embeddings

### 2. **Matrix Factorization**
- Decomposes user-item matrix into latent factors
- Implemented via embedding layers in TensorFlow

### 3. **Neural Collaborative Filtering (NeuMF)**
- **GMF Path**: Element-wise product of user/item embeddings
- **MLP Path**: Deep neural network for non-linear interactions
- **Fusion**: Combines both paths for final prediction

### 4. **Evaluation Metrics**
| Metric | Purpose |
|--------|---------|
| **MAE/MSE/RMSE** | Rating prediction accuracy |
| **Precision@K** | Relevance of top-K recommendations |
| **Recall@K** | Coverage of relevant items |
| **NDCG@K** | Ranking quality (position-aware) |
| **Hit Rate** | Success rate of recommendations |

---

## 📂 Project Structure

```
ShopSense/
├── app.py                 # Flask web application
├── verify_recommendations.py  # Verification script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
│
├── src/                   # Core modules
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Data loading & preprocessing
│   ├── models.py          # NeuMF & MF models
│   ├── evaluator.py       # MAE, MSE, RMSE metrics
│   ├── recommender.py     # Recommendation engine
│   └── verification.py    # Ranking metrics (Precision, Recall, NDCG)
│
├── templates/
│   └── index.html         # Web UI template
│
├── static/
│   ├── style.css          # Styling
│   └── script.js          # Frontend JavaScript
│
├── models/                # Saved models (generated)
│   ├── neumf_model.keras
│   ├── data_loader.pkl
│   └── metrics.pkl
│
└── product-recommendation-system.ipynb  # Jupyter notebook
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ShopSense.git
cd ShopSense

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the Amazon Electronics dataset:
```bash
# Option 1: Download from Kaggle
# https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews

# Option 2: Use any ratings dataset with columns:
# User_id, Product_id, rating, Timestamp
```

Place the CSV file in the project root as `ratings_Electronics (1).csv`

### Run the Application

```bash
python app.py
```

Open your browser at `http://localhost:5000`

---

## 💻 Usage

### 1. Train the Model
- Set **Sample Size** (10,000 - 50,000 records)
- Set **Epochs** (5-10 recommended)
- Click **Start Training**

### 2. Get Recommendations
- Enter a **User ID** or click **Random**
- Click **Get Recommendations**

### 3. Verify Recommendations
- Scroll to **Verify Recommendations**
- Click **Run Verification** for ranking metrics
- Click **Compare User** to see actual vs predicted

---

## 📊 Sample Results

### Model Performance
| Metric | Value |
|--------|-------|
| MAE | ~0.85 |
| MSE | ~1.12 |
| RMSE | ~1.06 |

### Ranking Metrics
| Metric | Value |
|--------|-------|
| Precision@10 | Varies with data |
| Hit Rate@10 | Varies with data |
| Catalog Coverage | ~10-15% |

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Data settings
SAMPLE_SIZE = 50000
TEST_SIZE = 0.2

# Model settings
LATENT_DIM = 10
EPOCHS = 10
BATCH_SIZE = 32
```

---

## 📚 References

1. **Neural Collaborative Filtering** - He et al., 2017
   - [Paper](https://arxiv.org/abs/1708.05031)
   
2. **Matrix Factorization Techniques for Recommender Systems**
   - Koren, Bell, Volinsky (Netflix Prize)

3. **Amazon Product Reviews Dataset**
   - [Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)

---

## 👤 Author

**Taha Zaidi (23K-0577)**  
**Shahmeer Irfan (23K-0832)**  
FAST-NUCES | Fall 2025  
Recommender Systems - Final Project

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for Recommender Systems Course
</p>
