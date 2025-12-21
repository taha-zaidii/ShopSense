"""
Recommendation Verification Module
Provides comprehensive metrics to verify recommendation quality against actual data.
"""
import numpy as np
import pandas as pd
from collections import defaultdict


class RecommendationVerifier:
    """
    Comprehensive verifier for recommendation systems.
    Computes ranking metrics to validate recommendations against ground truth.
    """
    
    def __init__(self, model, data_loader):
        """
        Initialize verifier.
        
        Args:
            model: Trained recommendation model
            data_loader: DataLoader instance with processed data
        """
        self.model = model
        self.data_loader = data_loader
        
    def get_user_ground_truth(self, user_id, threshold=4.0):
        """
        Get ground truth (actually liked/highly-rated) items for a user.
        
        Args:
            user_id: Numeric user ID
            threshold: Rating threshold to consider an item as "relevant"
            
        Returns:
            set: Set of product IDs that the user rated >= threshold
        """
        user_data = self.data_loader.dataset[
            self.data_loader.dataset.user_id_numeric == user_id
        ]
        relevant_items = user_data[user_data.rating >= threshold]['product_id_numeric'].values
        return set(relevant_items)
    
    def get_all_user_ratings(self, user_id):
        """
        Get all ratings for a user.
        
        Returns:
            dict: {product_id: rating}
        """
        user_data = self.data_loader.dataset[
            self.data_loader.dataset.user_id_numeric == user_id
        ]
        return dict(zip(user_data['product_id_numeric'], user_data['rating']))
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Precision@K.
        
        Precision@K = (# of recommended items @K that are relevant) / K
        
        Args:
            recommended_items: List of recommended product IDs (in ranked order)
            relevant_items: Set of relevant (ground truth) product IDs
            k: Number of top recommendations to consider
            
        Returns:
            float: Precision@K score
        """
        if k == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & relevant_items)
        return relevant_in_top_k / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Recall@K.
        
        Recall@K = (# of recommended items @K that are relevant) / (total # of relevant items)
        
        Args:
            recommended_items: List of recommended product IDs
            relevant_items: Set of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            float: Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommended_items[:k]
        relevant_in_top_k = len(set(top_k) & relevant_items)
        return relevant_in_top_k / len(relevant_items)
    
    def hit_rate_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Hit Rate@K (also called Recall@K binary).
        
        Hit@K = 1 if at least one relevant item in top K, else 0
        
        Args:
            recommended_items: List of recommended product IDs
            relevant_items: Set of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            int: 1 if hit, 0 otherwise
        """
        top_k = set(recommended_items[:k])
        return 1 if len(top_k & relevant_items) > 0 else 0
    
    def dcg_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Discounted Cumulative Gain at K.
        
        DCG@K = sum(rel_i / log2(i + 1)) for i from 1 to K
        
        Args:
            recommended_items: List of recommended product IDs
            relevant_items: Set of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            float: DCG@K score
        """
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                # Relevance is binary: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
        return dcg
    
    def ndcg_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        NDCG@K = DCG@K / IDCG@K
        
        Args:
            recommended_items: List of recommended product IDs
            relevant_items: Set of relevant product IDs
            k: Number of top recommendations to consider
            
        Returns:
            float: NDCG@K score
        """
        dcg = self.dcg_at_k(recommended_items, relevant_items, k)
        
        # Ideal DCG: all relevant items at top positions
        ideal_relevant_count = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant_count))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def mean_average_precision(self, recommended_items, relevant_items, k):
        """
        Calculate Average Precision at K.
        
        AP@K = (1/min(m,K)) * sum(P(i) * rel(i)) for i from 1 to K
        where m = # of relevant items
        
        Args:
            recommended_items: List of recommended product IDs
            relevant_items: Set of relevant product IDs
            k: Number of top recommendations
            
        Returns:
            float: Average Precision score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0
        
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                num_hits += 1
                score += num_hits / (i + 1)
        
        return score / min(len(relevant_items), k)
    
    def calculate_catalog_coverage(self, all_recommendations, total_products):
        """
        Calculate what percentage of products ever get recommended.
        
        Args:
            all_recommendations: Dict of {user_id: [recommended_product_ids]}
            total_products: Total number of products in catalog
            
        Returns:
            float: Coverage percentage (0-100)
        """
        recommended_products = set()
        for recs in all_recommendations.values():
            recommended_products.update(recs)
        
        return (len(recommended_products) / total_products) * 100
    
    def evaluate_user(self, user_id, k_values=[5, 10, 20], threshold=4.0):
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: Numeric user ID
            k_values: List of K values to evaluate at
            threshold: Rating threshold for relevance
            
        Returns:
            dict: Metrics for this user at each K
        """
        # Get ground truth (items user rated highly)
        relevant_items = self.get_user_ground_truth(user_id, threshold)
        
        if len(relevant_items) == 0:
            return None  # Skip users with no relevant items
        
        # Get products user has rated (to exclude from recommendations)
        rated_products = set(self.data_loader.dataset[
            self.data_loader.dataset.user_id_numeric == user_id
        ]['product_id_numeric'].values)
        
        # Get all candidate products (not rated by user)
        all_products = self.data_loader.dataset['product_id_numeric'].unique()
        candidate_products = [p for p in all_products if p not in rated_products]
        
        if len(candidate_products) == 0:
            return None
        
        # Limit for speed
        candidate_products = candidate_products[:min(500, len(candidate_products))]
        
        # Generate predictions
        user_arr = np.array([user_id] * len(candidate_products))
        prod_arr = np.array(candidate_products)
        
        predictions = self.model.predict(user_arr, prod_arr).flatten()
        
        # Sort by predicted score
        sorted_indices = np.argsort(-predictions)
        recommended_items = [candidate_products[i] for i in sorted_indices]
        
        # Calculate metrics at each K
        metrics = {}
        for k in k_values:
            metrics[f'Precision@{k}'] = self.precision_at_k(recommended_items, relevant_items, k)
            metrics[f'Recall@{k}'] = self.recall_at_k(recommended_items, relevant_items, k)
            metrics[f'Hit@{k}'] = self.hit_rate_at_k(recommended_items, relevant_items, k)
            metrics[f'NDCG@{k}'] = self.ndcg_at_k(recommended_items, relevant_items, k)
            metrics[f'MAP@{k}'] = self.mean_average_precision(recommended_items, relevant_items, k)
        
        metrics['num_relevant'] = len(relevant_items)
        metrics['recommended_items'] = recommended_items[:max(k_values)]
        
        return metrics
    
    def evaluate_all_users(self, sample_users=100, k_values=[5, 10, 20], threshold=4.0, verbose=True):
        """
        Evaluate recommendations for multiple users and aggregate metrics.
        
        Args:
            sample_users: Number of users to sample (None for all)
            k_values: List of K values to evaluate at
            threshold: Rating threshold for relevance
            verbose: Print progress
            
        Returns:
            dict: Aggregated metrics across all users
        """
        # Get unique users
        unique_users = self.data_loader.dataset['user_id_numeric'].unique()
        
        # Sample users if needed
        if sample_users and len(unique_users) > sample_users:
            user_sample = np.random.choice(unique_users, sample_users, replace=False)
        else:
            user_sample = unique_users
        
        # Collect metrics
        all_metrics = defaultdict(list)
        evaluated_users = 0
        all_recommendations = {}
        
        for i, user_id in enumerate(user_sample):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluating user {i+1}/{len(user_sample)}...")
            
            user_metrics = self.evaluate_user(user_id, k_values, threshold)
            
            if user_metrics is None:
                continue
            
            evaluated_users += 1
            all_recommendations[user_id] = user_metrics.pop('recommended_items')
            
            for metric_name, value in user_metrics.items():
                all_metrics[metric_name].append(value)
        
        if evaluated_users == 0:
            return {'error': 'No users could be evaluated'}
        
        # Aggregate metrics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if metric_name == 'num_relevant':
                aggregated[metric_name] = np.mean(values)
            else:
                aggregated[metric_name] = np.mean(values)
        
        aggregated['evaluated_users'] = evaluated_users
        aggregated['total_sampled'] = len(user_sample)
        
        # Calculate catalog coverage
        total_products = self.data_loader.num_products
        aggregated['catalog_coverage'] = self.calculate_catalog_coverage(
            all_recommendations, total_products
        )
        
        return aggregated
    
    def leave_one_out_evaluation(self, sample_users=100, k_values=[5, 10, 20], verbose=True):
        """
        Leave-one-out evaluation: For each user, hold out their highest-rated item
        and check if the model recommends it.
        
        Args:
            sample_users: Number of users to sample
            k_values: List of K values for Hit@K
            verbose: Print progress
            
        Returns:
            dict: Hit rates at each K
        """
        unique_users = self.data_loader.dataset['user_id_numeric'].unique()
        
        if sample_users and len(unique_users) > sample_users:
            user_sample = np.random.choice(unique_users, sample_users, replace=False)
        else:
            user_sample = unique_users
        
        hits = {k: 0 for k in k_values}
        evaluated_users = 0
        
        for i, user_id in enumerate(user_sample):
            if verbose and (i + 1) % 10 == 0:
                print(f"  LOO Evaluation: user {i+1}/{len(user_sample)}...")
            
            # Get user's ratings
            user_data = self.data_loader.dataset[
                self.data_loader.dataset.user_id_numeric == user_id
            ]
            
            if len(user_data) < 2:
                continue
            
            # Get highest-rated item (ground truth)
            highest_rated_idx = user_data['rating'].idxmax()
            held_out_item = user_data.loc[highest_rated_idx, 'product_id_numeric']
            
            # Get other rated items
            other_rated = set(user_data['product_id_numeric'].values) - {held_out_item}
            
            # Get candidates (all products - other rated)
            all_products = self.data_loader.dataset['product_id_numeric'].unique()
            candidates = [p for p in all_products if p not in other_rated]
            
            if len(candidates) == 0:
                continue
            
            # Limit for speed
            candidates = candidates[:min(500, len(candidates))]
            
            # Make sure held_out_item is in candidates
            if held_out_item not in candidates:
                candidates = [held_out_item] + candidates[:499]
            
            # Predict
            user_arr = np.array([user_id] * len(candidates))
            prod_arr = np.array(candidates)
            
            predictions = self.model.predict(user_arr, prod_arr).flatten()
            
            # Rank
            sorted_indices = np.argsort(-predictions)
            recommended = [candidates[i] for i in sorted_indices]
            
            evaluated_users += 1
            
            # Check hits at each K
            for k in k_values:
                if held_out_item in recommended[:k]:
                    hits[k] += 1
        
        if evaluated_users == 0:
            return {'error': 'No users could be evaluated'}
        
        results = {
            f'LOO_Hit@{k}': hits[k] / evaluated_users for k in k_values
        }
        results['evaluated_users'] = evaluated_users
        
        return results
    
    def compare_with_actual_ratings(self, user_id, top_n=10):
        """
        Compare recommendations with user's actual rating history.
        
        Args:
            user_id: Numeric user ID
            top_n: Number of recommendations to analyze
            
        Returns:
            dict: Comparison data including user's history and recommendations
        """
        # Get user's actual ratings
        user_data = self.data_loader.dataset[
            self.data_loader.dataset.user_id_numeric == user_id
        ].sort_values('rating', ascending=False)
        
        actual_history = [
            {
                'product_id': int(row['product_id_numeric']),
                'rating': float(row['rating'])
            }
            for _, row in user_data.iterrows()
        ]
        
        # Get recommendations
        rated_products = set(user_data['product_id_numeric'].values)
        all_products = self.data_loader.dataset['product_id_numeric'].unique()
        candidates = [p for p in all_products if p not in rated_products][:200]
        
        if len(candidates) == 0:
            return {'error': 'No candidates available'}
        
        user_arr = np.array([user_id] * len(candidates))
        prod_arr = np.array(candidates)
        
        predictions = self.model.predict(user_arr, prod_arr).flatten()
        
        recs = sorted(zip(candidates, predictions), key=lambda x: -x[1])[:top_n]
        
        recommendations = [
            {
                'product_id': int(prod),
                'predicted_rating': float(pred)
            }
            for prod, pred in recs
        ]
        
        # Calculate user's average rating
        avg_rating = user_data['rating'].mean()
        high_rated_count = len(user_data[user_data['rating'] >= 4])
        
        return {
            'user_id': user_id,
            'total_ratings': len(user_data),
            'avg_rating': float(avg_rating),
            'high_rated_items': high_rated_count,
            'actual_top_rated': actual_history[:10],
            'recommendations': recommendations,
            'predicted_avg_rating': np.mean([r['predicted_rating'] for r in recommendations])
        }


def print_verification_report(metrics, title="Recommendation Verification Report"):
    """
    Print a formatted verification report.
    
    Args:
        metrics: Dict of metrics from evaluate_all_users or leave_one_out_evaluation
        title: Report title
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    
    if 'error' in metrics:
        print(f"\n❌ Error: {metrics['error']}")
        return
    
    print(f"\n📊 Evaluation Summary:")
    print(f"   • Users Evaluated: {metrics.get('evaluated_users', 'N/A')}")
    print(f"   • Avg Relevant Items per User: {metrics.get('num_relevant', 0):.2f}")
    print(f"   • Catalog Coverage: {metrics.get('catalog_coverage', 0):.2f}%")
    
    print(f"\n📈 Ranking Metrics:")
    
    # Group metrics by K value
    k_values = set()
    for key in metrics.keys():
        if '@' in key:
            k = int(key.split('@')[1])
            k_values.add(k)
    
    for k in sorted(k_values):
        print(f"\n   At K = {k}:")
        if f'Precision@{k}' in metrics:
            print(f"      • Precision@{k}:  {metrics[f'Precision@{k}']:.4f}")
        if f'Recall@{k}' in metrics:
            print(f"      • Recall@{k}:     {metrics[f'Recall@{k}']:.4f}")
        if f'Hit@{k}' in metrics:
            print(f"      • Hit Rate@{k}:   {metrics[f'Hit@{k}']:.4f}")
        if f'NDCG@{k}' in metrics:
            print(f"      • NDCG@{k}:       {metrics[f'NDCG@{k}']:.4f}")
        if f'MAP@{k}' in metrics:
            print(f"      • MAP@{k}:        {metrics[f'MAP@{k}']:.4f}")
        if f'LOO_Hit@{k}' in metrics:
            print(f"      • LOO Hit@{k}:    {metrics[f'LOO_Hit@{k}']:.4f}")
    
    print("\n" + "=" * 70)
    print(" Metric Interpretations:")
    print("-" * 70)
    print(" • Precision@K: What % of recommended items are actually relevant?")
    print(" • Recall@K: What % of relevant items appear in recommendations?")
    print(" • Hit Rate@K: Does at least 1 relevant item appear in top K?")
    print(" • NDCG@K: Are relevant items ranked higher? (considers position)")
    print(" • MAP@K: Mean Average Precision - overall ranking quality")
    print(" • LOO Hit@K: Can model predict user's highest-rated item?")
    print("=" * 70 + "\n")
