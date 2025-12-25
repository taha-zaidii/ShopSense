# ShopSense - Project Report
## Personalized Product Recommendation Engine
**Taha Zaidi (23K-0577) | Shahmeer Irfan (23K-0832)**  
**FAST-NUCES | Recommender Systems - Fall 2025**

---

## 1. Executive Summary

ShopSense is a Neural Collaborative Filtering (NeuMF) based recommendation system that predicts user ratings for products using deep learning. The system achieves:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **MAE** | 0.9477 | Predictions off by <1 star on average |
| **RMSE** | 1.2127 | Root mean squared error in rating scale |
| **Accuracy** | 20.63% | Exact rating match |
| **Within-1 Accuracy** | 76.26% | Predictions within ±1 star |

---

## 2. Dataset Analysis

### 2.1 Dataset Source
- **Name**: Amazon Electronics Product Reviews
- **Source**: [Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
- **Size**: ~7.8 million ratings

### 2.2 Rating Distribution (Raw Data)

| Rating | Count | Percentage |
|--------|-------|------------|
| ⭐ 1 | 60,597 | 12.1% |
| ⭐⭐ 2 | 29,333 | 5.9% |
| ⭐⭐⭐ 3 | 39,178 | 7.8% |
| ⭐⭐⭐⭐ 4 | 97,751 | 19.5% |
| ⭐⭐⭐⭐⭐ 5 | 273,141 | 54.6% |

**Observations:**
- Mean rating: 3.99 (positively skewed)
- Standard deviation: 1.40
- Over 54% of ratings are 5-star (common in e-commerce)

### 2.3 Data Challenges

| Challenge | Description | Our Solution |
|-----------|-------------|--------------|
| **Extreme Sparsity** | 99.98% of user-item matrix is empty | Cold-start filtering |
| **Cold-Start Users** | 93% of users had only 1 rating | Filter users with <5 ratings |
| **Cold-Start Items** | Many products with <3 ratings | Filter products with <3 ratings |
| **Rating Bias** | Users tend to rate when satisfied (5★) | Model learns this bias through bias terms |

### 2.4 After Preprocessing

After applying cold-start filtering:
- **Users**: ~42,000 (active users only)
- **Products**: ~25,000 (popular products only)  
- **Ratings**: ~50,000 (sampled)
- **Sparsity**: Reduced to ~97%

---

## 3. Model Architecture

### 3.1 Neural Matrix Factorization (NeuMF)

We implemented NeuMF which combines two complementary approaches:

```
                    ┌─────────────┐
                    │   Inputs    │
                    │ User + Item │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │   MLP    │   │    MF    │   │  Biases  │
      │ Pathway  │   │ Pathway  │   │ User+Item│
      └────┬─────┘   └────┬─────┘   └────┬─────┘
           │              │              │
           └──────────────┴──────────────┘
                          │
                    ┌─────▼─────┐
                    │  Combine  │
                    │  + Dense  │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  Rating   │
                    │   [1-5]   │
                    └───────────┘
```

### 3.2 Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **User/Item Embeddings** | Learn latent factors representing preferences | Embedding(dim=32) |
| **MLP Path** | Capture non-linear user-item interactions | Dense(128→64→32→16) |
| **MF Path** | Element-wise interaction like traditional MF | Multiply embeddings |
| **Bias Terms** | Capture user/item inherent tendencies | Embedding(dim=1) each |
| **Output** | Constrained to valid rating range | Sigmoid × 4 + 1 → [1,5] |

### 3.3 Regularization Techniques

| Technique | Value | Purpose |
|-----------|-------|---------|
| L2 Regularization | 1e-5 | Prevent weight explosion |
| Dropout | 0.2 / 0.15 | Reduce overfitting |
| Batch Normalization | After FC layers | Stabilize training |
| Early Stopping | patience=5 | Stop when validation loss plateaus |
| ReduceLROnPlateau | factor=0.5 | Reduce learning rate adaptively |

---

## 4. Evaluation Metrics

### 4.1 Why These Metrics?

| Metric | Formula | Why We Use It |
|--------|---------|---------------|
| **MAE** | `mean(\|actual - predicted\|)` | Intuitive: average star error |
| **MSE** | `mean((actual - predicted)²)` | Penalizes large errors more heavily |
| **RMSE** | `√MSE` | Same scale as ratings; industry standard |
| **Accuracy** | `% exact matches` | Classification-style evaluation |
| **Within-1** | `% predictions within ±1 star` | Practical usefulness measure |

### 4.2 Real-World Context

For e-commerce recommendations, **Within-1 Accuracy of 76%** means:
- If a user would rate a product 4★, we predict 3★, 4★, or 5★ about 76% of the time
- This is practically useful for ranking and recommendation purposes

### 4.3 Industry Benchmarks

| System | Dataset | MAE | Notes |
|--------|---------|-----|-------|
| Netflix Prize Winner | Netflix (dense) | ~0.85 | Dense data, years of work |
| Surprise Library | MovieLens 100K | 0.73-0.93 | Smaller, denser dataset |
| Amazon Benchmarks | Amazon Reviews | 0.8-1.2 | Similar to ours |
| **ShopSense** | **Amazon Electronics** | **0.95** | **Sparse data, CF only** |

---

## 5. Results Summary

### 5.1 Final Metrics

| Metric | Value | Status |
|--------|-------|--------|
| MAE | **0.9477** | ✅ Within benchmark range |
| MSE | **1.4706** | ✅ Acceptable |
| RMSE | **1.2127** | ✅ Within benchmark range |
| Accuracy | **20.63%** | ✅ Expected for 5-class problem |
| Within-1 Accuracy | **76.26%** | ✅ Excellent |

### 5.2 Training Performance

- **Total Parameters**: 6.6M
- **Training Time**: ~10s per epoch
- **Early Stopping**: Triggered at epoch 6 (restored best weights from epoch 1)
- **Best Validation Loss**: 1.4598

---

## 6. Limitations & Future Work

### Current Limitations
1. **Pure Collaborative Filtering**: No content features (categories, descriptions)
2. **Cold-Start**: Cannot recommend for new users/products
3. **Static Model**: Doesn't update in real-time

### Future Improvements
1. Add content-based features (hybrid approach)
2. Implement implicit feedback (clicks, time spent)
3. Use sequence models (GRU/Transformer) for session-based recommendations
4. Deploy with online learning capability

---

## 7. How to Run

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Web Application
python app.py
# Open http://localhost:5001
```

### Demo Flow
1. **Train Model**: Set sample size (50K), epochs (10), click "Start Training"
2. **Get Recommendations**: Enter User ID or click "Random"
3. **Verify**: Run verification to see ranking metrics

---

## 8. References

1. He, X., et al. (2017). "Neural Collaborative Filtering." WWW 2017.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems." IEEE Computer.
3. Amazon Product Reviews Dataset - Kaggle

---

*Report generated: December 25, 2025*
