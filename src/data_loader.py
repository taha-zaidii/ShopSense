"""
Data loading and preprocessing module for the product recommendation system.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handles data loading and preprocessing for the recommendation system."""
    
    def __init__(self, file_path, sample_size=50000, min_user_ratings=5, min_product_ratings=3):
        """
        Initialize DataLoader.
        
        Args:
            file_path (str): Path to the CSV file
            sample_size (int): Number of samples to use (default: 50000)
            min_user_ratings (int): Minimum ratings per user to keep (default: 5)
            min_product_ratings (int): Minimum ratings per product to keep (default: 3)
        """
        self.file_path = file_path
        self.sample_size = sample_size
        self.min_user_ratings = min_user_ratings
        self.min_product_ratings = min_product_ratings
        self.dataset = None
        self.user_id_to_numeric = None
        self.product_id_to_numeric = None
        self.num_users = None
        self.num_products = None
        self.sparsity = None
        self.global_mean = None
        
    def load_data(self):
        """Load the dataset from CSV file."""
        print(f"Loading data from {self.file_path}...")
        self.dataset = pd.read_csv(self.file_path)
        self.dataset.columns = ['User_id', 'Product_id', 'rating', 'Timestamp']
        print(f"Data loaded successfully. Shape: {self.dataset.shape}")
        return self
    
    def _filter_cold_start(self):
        """Filter out users and products with too few ratings (cold-start problem)."""
        original_size = len(self.dataset)
        
        # Iteratively filter until stable (filtering users may create sparse products and vice versa)
        prev_size = 0
        iteration = 0
        while prev_size != len(self.dataset) and iteration < 5:
            prev_size = len(self.dataset)
            iteration += 1
            
            # Filter users with minimum ratings
            user_counts = self.dataset['User_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            self.dataset = self.dataset[self.dataset['User_id'].isin(valid_users)]
            
            # Filter products with minimum ratings
            product_counts = self.dataset['Product_id'].value_counts()
            valid_products = product_counts[product_counts >= self.min_product_ratings].index
            self.dataset = self.dataset[self.dataset['Product_id'].isin(valid_products)]
        
        filtered_size = len(self.dataset)
        print(f"Cold-start filtering: {original_size:,} → {filtered_size:,} ratings "
              f"(removed {original_size - filtered_size:,} sparse interactions)")
        
    def preprocess_data(self):
        """Preprocess the data by filtering cold-start and converting IDs to numeric format."""
        print("Preprocessing data...")
        
        # Store global mean before any filtering (for bias calculation)
        self.global_mean = self.dataset['rating'].mean()
        
        # Filter cold-start users and products FIRST
        self._filter_cold_start()
        
        # Random sample if dataset is still too large (stratified by rating)
        if self.sample_size and len(self.dataset) > self.sample_size:
            # Stratified sampling to preserve rating distribution
            self.dataset = self.dataset.groupby('rating', group_keys=False).apply(
                lambda x: x.sample(frac=self.sample_size/len(self.dataset), random_state=42)
            ).reset_index(drop=True)
            print(f"Stratified sampling to ~{len(self.dataset):,} samples")
        
        # Convert user IDs to numeric
        unique_user_ids = self.dataset['User_id'].unique()
        self.user_id_to_numeric = {user_id: idx for idx, user_id in enumerate(unique_user_ids, start=1)}
        self.dataset['user_id_numeric'] = self.dataset['User_id'].map(self.user_id_to_numeric)
        
        # Convert product IDs to numeric
        unique_product_ids = self.dataset['Product_id'].unique()
        self.product_id_to_numeric = {product_id: idx for idx, product_id in enumerate(unique_product_ids, start=1)}
        self.dataset['product_id_numeric'] = self.dataset['Product_id'].map(self.product_id_to_numeric)
        
        # Drop original ID columns
        self.dataset.drop(['User_id', 'Product_id'], axis=1, inplace=True)
        
        # Reorder columns
        self.dataset = self.dataset[['user_id_numeric', 'product_id_numeric', 'rating', 'Timestamp']]
        
        self.num_users = len(self.dataset.user_id_numeric.unique())
        self.num_products = len(self.dataset.product_id_numeric.unique())
        
        # Calculate sparsity
        self.sparsity = 1 - (len(self.dataset) / (self.num_users * self.num_products))
        
        print(f"Preprocessing complete:")
        print(f"  Users: {self.num_users:,}, Products: {self.num_products:,}")
        print(f"  Ratings: {len(self.dataset):,}, Sparsity: {self.sparsity*100:.2f}%")
        print(f"  Global mean rating: {self.global_mean:.2f}")
        return self
        
    def split_data(self, test_size=0.2, random_state=None):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (train_data, test_data)
        """
        print(f"Splitting data with test_size={test_size}...")
        train, test = train_test_split(self.dataset, test_size=test_size, random_state=random_state)
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        return train, test
        
    def get_data_info(self):
        """Return information about the loaded dataset."""
        return {
            'num_users': self.num_users,
            'num_products': self.num_products,
            'dataset_shape': self.dataset.shape,
            'user_id_mapping': self.user_id_to_numeric,
            'product_id_mapping': self.product_id_to_numeric
        }
