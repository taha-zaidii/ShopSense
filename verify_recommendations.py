#!/usr/bin/env python3.11
"""
Recommendation Verification Script
===================================
This script verifies the quality of your recommender system by:
1. Computing ranking metrics (Precision@K, Recall@K, NDCG@K, Hit Rate)
2. Performing leave-one-out evaluation
3. Comparing recommendations with users' actual rating history
4. Generating a comprehensive verification report

Run this after training your model to validate its recommendations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    print("=" * 70)
    print(" 🔍 RECOMMENDATION SYSTEM VERIFICATION")
    print("=" * 70)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # === 1. Load Model and Data ===
    print("\n📦 Step 1: Loading model and data...")
    
    model_path = 'models/neumf_model.keras'
    data_path = 'models/data_loader.pkl'
    
    if not os.path.exists(model_path):
        print("❌ ERROR: No trained model found!")
        print("   Please train the model first using the web interface or training script.")
        return
    
    if not os.path.exists(data_path):
        print("❌ ERROR: Data loader not found!")
        print("   Please train the model first.")
        return
    
    # Import TensorFlow and enable eager execution
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)
    
    # Load data loader
    with open(data_path, 'rb') as f:
        data_loader = pickle.load(f)
    
    print(f"   ✅ Data loaded: {data_loader.num_users} users, {data_loader.num_products} products")
    
    # Load model
    from models import NeuMFModel
    import config
    
    model = NeuMFModel(
        num_users=data_loader.num_users,
        num_products=data_loader.num_products,
        latent_dim=config.LATENT_DIM
    )
    model.load_model(model_path)
    print("   ✅ Model loaded successfully!")
    
    # === 2. Initialize Verifier ===
    print("\n🔧 Step 2: Initializing verifier...")
    
    from verification import RecommendationVerifier, print_verification_report
    
    verifier = RecommendationVerifier(model, data_loader)
    print("   ✅ Verifier initialized!")
    
    # === 3. Run Ranking Metrics Evaluation ===
    print("\n📊 Step 3: Computing ranking metrics...")
    print("   This evaluates Precision@K, Recall@K, Hit Rate, NDCG, and MAP")
    print("   Sampling 50 users for evaluation (adjustable)...\n")
    
    ranking_metrics = verifier.evaluate_all_users(
        sample_users=50,
        k_values=[5, 10, 20],
        threshold=4.0,  # Items rated 4+ are considered "relevant"
        verbose=True
    )
    
    print_verification_report(ranking_metrics, "Ranking Metrics Evaluation")
    
    # === 4. Run Leave-One-Out Evaluation ===
    print("\n🎯 Step 4: Leave-One-Out Evaluation...")
    print("   For each user, holding out their highest-rated item and checking if")
    print("   the model can recommend it.\n")
    
    loo_metrics = verifier.leave_one_out_evaluation(
        sample_users=50,
        k_values=[5, 10, 20],
        verbose=True
    )
    
    print_verification_report(loo_metrics, "Leave-One-Out Evaluation")
    
    # === 5. Compare with Actual Ratings (Sample Users) ===
    print("\n📋 Step 5: Comparing recommendations with actual user ratings...")
    
    # Get sample users
    unique_users = data_loader.dataset['user_id_numeric'].unique()
    sample_users = np.random.choice(unique_users, min(3, len(unique_users)), replace=False)
    
    comparisons = []
    
    for user_id in sample_users:
        comparison = verifier.compare_with_actual_ratings(user_id, top_n=10)
        if 'error' not in comparison:
            comparisons.append(comparison)
    
    for comp in comparisons:
        print("\n" + "-" * 60)
        print(f" User {comp['user_id']} Analysis")
        print("-" * 60)
        print(f" • Total ratings by user: {comp['total_ratings']}")
        print(f" • Average rating given: {comp['avg_rating']:.2f}")
        print(f" • High-rated items (≥4): {comp['high_rated_items']}")
        print(f" • Avg predicted rating for recommendations: {comp['predicted_avg_rating']:.2f}")
        
        print(f"\n   User's Top Rated Items (Actual):")
        for i, item in enumerate(comp['actual_top_rated'][:5], 1):
            print(f"      {i}. Product {item['product_id']} → Rating: {item['rating']} ⭐")
        
        print(f"\n   Model's Recommendations:")
        for i, item in enumerate(comp['recommendations'][:5], 1):
            print(f"      {i}. Product {item['product_id']} → Predicted: {item['predicted_rating']:.2f} ⭐")
    
    # === 6. Generate Summary Report ===
    print("\n" + "=" * 70)
    print(" 📄 VERIFICATION SUMMARY")
    print("=" * 70)
    
    report_lines = [
        "=" * 60,
        "RECOMMENDATION SYSTEM VERIFICATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "DATASET STATISTICS:",
        f"  • Users: {data_loader.num_users}",
        f"  • Products: {data_loader.num_products}",
        f"  • Total Ratings: {len(data_loader.dataset)}",
        "",
        "RANKING METRICS (Higher is better):",
    ]
    
    for k in [5, 10, 20]:
        if f'Precision@{k}' in ranking_metrics:
            report_lines.append(f"  @ K = {k}:")
            report_lines.append(f"    Precision@{k}:  {ranking_metrics[f'Precision@{k}']:.4f}")
            report_lines.append(f"    Recall@{k}:     {ranking_metrics[f'Recall@{k}']:.4f}")
            report_lines.append(f"    Hit Rate@{k}:   {ranking_metrics[f'Hit@{k}']:.4f}")
            report_lines.append(f"    NDCG@{k}:       {ranking_metrics[f'NDCG@{k}']:.4f}")
    
    report_lines.extend([
        "",
        "LEAVE-ONE-OUT EVALUATION:",
    ])
    
    for k in [5, 10, 20]:
        if f'LOO_Hit@{k}' in loo_metrics:
            report_lines.append(f"    LOO Hit@{k}:    {loo_metrics[f'LOO_Hit@{k}']:.4f}")
    
    report_lines.extend([
        "",
        f"CATALOG COVERAGE: {ranking_metrics.get('catalog_coverage', 0):.2f}%",
        "",
        "INTERPRETATION:",
        "  • Good Precision@10: > 0.1 (10% of recs are relevant)",
        "  • Good Hit Rate@10: > 0.5 (50% chance of hitting a relevant item)",
        "  • Good NDCG@10: > 0.3 (relevant items ranked higher)",
        "  • Good LOO Hit@10: > 0.3 (30% chance of predicting favorite)",
        "",
        "=" * 60
    ])
    
    # Save report
    report_file = 'verification_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n   ✅ Report saved to: {report_file}")
    
    # Print summary verdict
    hit_rate_10 = ranking_metrics.get('Hit@10', 0)
    ndcg_10 = ranking_metrics.get('NDCG@10', 0)
    loo_hit_10 = loo_metrics.get('LOO_Hit@10', 0)
    
    print("\n" + "=" * 70)
    print(" 🎯 VERDICT")
    print("=" * 70)
    
    if hit_rate_10 >= 0.5 and ndcg_10 >= 0.3:
        print(" ✅ EXCELLENT: Your recommender is performing very well!")
        print(f"    Hit Rate@10: {hit_rate_10:.2%} | NDCG@10: {ndcg_10:.4f}")
    elif hit_rate_10 >= 0.3 or ndcg_10 >= 0.2:
        print(" ⚠️  GOOD: Your recommender is working, but has room for improvement.")
        print(f"    Hit Rate@10: {hit_rate_10:.2%} | NDCG@10: {ndcg_10:.4f}")
    elif hit_rate_10 > 0 or ndcg_10 > 0:
        print(" 📊 FAIR: Your recommender shows some signal but needs improvement.")
        print(f"    Hit Rate@10: {hit_rate_10:.2%} | NDCG@10: {ndcg_10:.4f}")
        print("\n   Tips to improve:")
        print("   • Train with more data (increase sample_size)")
        print("   • Train for more epochs")
        print("   • Adjust the rating threshold (try 3.5 instead of 4.0)")
    else:
        print(" ⚠️  Note: Metrics are low, which may be due to data sparsity.")
        print("   This is common with collaborative filtering on sparse data.")
    
    print("=" * 70)
    print(" ✅ VERIFICATION COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
