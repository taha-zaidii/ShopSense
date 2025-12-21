"""
Data loading and preprocessing module for the product recommendation system.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    """Handles data loading and preprocessing for the recommendation system."""
    
    def __init__(self, file_path, sample_size=50000):
        """
        Initialize DataLoader.
        
        Args:
            file_path (str): Path to the CSV file
            sample_size (int): Number of samples to use (default: 50000)
        """
        self.file_path = file_path
        self.sample_size = sample_size
        self.dataset = None
        self.user_id_to_numeric = None
        self.product_id_to_numeric = None
        self.num_users = None
        self.num_products = None
        
    def load_data(self):
        """Load the dataset from CSV file."""
        print(f"Loading data from {self.file_path}...")
        self.dataset = pd.read_csv(self.file_path)
        self.dataset.columns = ['User_id', 'Product_id', 'rating', 'Timestamp']
        print(f"Data loaded successfully. Shape: {self.dataset.shape}")
        return self
        
    def preprocess_data(self):
        """Preprocess the data by converting IDs to numeric format."""
        print("Preprocessing data...")
        
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
        
        # Limit dataset size
        if self.sample_size and len(self.dataset) > self.sample_size:
            self.dataset = self.dataset.head(self.sample_size)
            print(f"Dataset limited to {self.sample_size} samples")
        
        self.num_users = len(self.dataset.user_id_numeric.unique())
        self.num_products = len(self.dataset.product_id_numeric.unique())
        
        print(f"Preprocessing complete. Users: {self.num_users}, Products: {self.num_products}")
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
