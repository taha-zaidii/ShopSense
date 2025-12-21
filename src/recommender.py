"""
Recommendation system that provides product recommendations to users.
"""
import numpy as np
import pandas as pd


class ProductRecommender:
    """High-level interface for making product recommendations."""
    
    def __init__(self, model, data_loader):
        """
        Initialize recommender.
        
        Args:
            model: Trained recommendation model
            data_loader: DataLoader instance with processed data
        """
        self.model = model
        self.data_loader = data_loader
        
    def recommend_products(self, user_id, top_n=10, exclude_rated=True):
        """
        Recommend top N products for a user.
        
        Args:
            user_id (int): Numeric user ID
            top_n (int): Number of recommendations to return
            exclude_rated (bool): Whether to exclude already rated products
            
        Returns:
            list: List of (product_id, predicted_rating) tuples
        """
        # Get all product IDs
        all_product_ids = self.data_loader.dataset.product_id_numeric.unique()
        
        if exclude_rated:
            # Get products already rated by user
            rated_products = self.data_loader.dataset[
                self.data_loader.dataset.user_id_numeric == user_id
            ].product_id_numeric.values
            
            # Filter out rated products
            candidate_products = [p for p in all_product_ids if p not in rated_products]
        else:
            candidate_products = all_product_ids
        
        # Prepare data for prediction
        user_ids = np.array([user_id] * len(candidate_products))
        product_ids = np.array(candidate_products)
        
        # Make predictions
        predictions = self.model.predict(user_ids, product_ids, round_decimals=2)
        
        # Combine products with predictions
        recommendations = list(zip(candidate_products, predictions.flatten()))
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return recommendations[:top_n]
        
    def recommend_for_multiple_users(self, user_ids, top_n=10):
        """
        Get recommendations for multiple users.
        
        Args:
            user_ids (list): List of user IDs
            top_n (int): Number of recommendations per user
            
        Returns:
            dict: Dictionary mapping user_id to list of recommendations
        """
        recommendations = {}
        
        for user_id in user_ids:
            recommendations[user_id] = self.recommend_products(user_id, top_n)
            
        return recommendations
        
    def get_similar_products(self, product_id, top_n=10):
        """
        Find products similar to a given product based on user ratings.
        
        Args:
            product_id (int): Product ID to find similar products for
            top_n (int): Number of similar products to return
            
        Returns:
            list: List of similar product IDs
        """
        # Get users who rated this product
        users_who_rated = self.data_loader.dataset[
            self.data_loader.dataset.product_id_numeric == product_id
        ].user_id_numeric.values
        
        if len(users_who_rated) == 0:
            return []
        
        # Get products rated by these users
        similar_products = self.data_loader.dataset[
            self.data_loader.dataset.user_id_numeric.isin(users_who_rated)
        ].product_id_numeric.value_counts()
        
        # Remove the original product
        similar_products = similar_products[similar_products.index != product_id]
        
        # Return top N
        return similar_products.head(top_n).index.tolist()
