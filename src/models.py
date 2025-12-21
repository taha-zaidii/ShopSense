"""
Neural Network models for product recommendation.
"""
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Embedding, Flatten, Input, Dropout, Dense, 
    BatchNormalization, Concatenate, Multiply, Lambda, Dot
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import tensorflow as tf


class MatrixFactorizationModel:
    """Matrix Factorization based recommendation model."""
    
    def __init__(self, num_users, num_products, latent_dim=10):
        """
        Initialize Matrix Factorization model.
        
        Args:
            num_users (int): Number of unique users
            num_products (int): Number of unique products
            latent_dim (int): Dimension of latent factors
        """
        self.num_users = num_users
        self.num_products = num_products
        self.latent_dim = latent_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the Matrix Factorization model architecture."""
        print("Building Matrix Factorization model...")
        
        # Input layers with explicit shape
        product_input = Input(shape=(1,), dtype='int32', name='product-input')
        user_input = Input(shape=(1,), dtype='int32', name='user-input')
        
        # Embedding layers
        product_embedding = Embedding(
            self.num_products + 1, 
            self.latent_dim, 
            name='product-embedding'
        )(product_input)
        user_embedding = Embedding(
            self.num_users + 1, 
            self.latent_dim, 
            name='user-embedding'
        )(user_input)
        
        # Flatten embeddings
        product_vec = Flatten(name='product-flatten')(product_embedding)
        user_vec = Flatten(name='user-flatten')(user_embedding)
        
        # Dot product of user and product vectors
        prod = Dot(axes=1, name='dot-product')([product_vec, user_vec])
        
        # Define model
        self.model = Model([user_input, product_input], prod)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Matrix Factorization model built successfully")
        self.model.summary()
        return self
        
    def train(self, train_data, epochs=10, batch_size=32):
        """
        Train the model.
        
        Args:
            train_data (DataFrame): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print(f"Training Matrix Factorization model for {epochs} epochs...")
        self.history = self.model.fit(
            [train_data.user_id_numeric, train_data.product_id_numeric],
            train_data.rating,
            epochs=epochs,
            batch_size=batch_size
        )
        return self.history
        
    def predict(self, user_ids, product_ids, round_decimals=2):
        """
        Make predictions.
        
        Args:
            user_ids: User IDs for prediction
            product_ids: Product IDs for prediction
            round_decimals (int): Number of decimals to round predictions
            
        Returns:
            Predicted ratings
        """
        predictions = self.model.predict([user_ids, product_ids])
        return np.round(predictions, decimals=round_decimals)
    
    def save_model(self, filepath):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """Load a saved model from disk."""
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")


class NeuMFModel:
    """Neural Matrix Factorization (NeuMF) recommendation model."""
    
    def __init__(self, num_users, num_products, latent_dim=10):
        """
        Initialize NeuMF model.
        
        Args:
            num_users (int): Number of unique users
            num_products (int): Number of unique products
            latent_dim (int): Dimension of latent factors
        """
        self.num_users = num_users
        self.num_products = num_products
        self.latent_dim = latent_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the NeuMF model architecture combining MLP and MF."""
        print("Building NeuMF model...")
        
        # Input layers with explicit shape
        product_input = Input(shape=(1,), dtype='int32', name='product-input')
        user_input = Input(shape=(1,), dtype='int32', name='user-input')
        
        # ===== MLP Path =====
        # MLP Embeddings with regularization
        product_embedding_mlp = Embedding(
            self.num_products + 1, 
            self.latent_dim * 2,  # Larger embeddings for MLP
            embeddings_regularizer=l2(1e-6),
            name='product-embedding-mlp'
        )(product_input)
        product_vec_mlp = Flatten(name='flatten-product-mlp')(product_embedding_mlp)
        
        user_embedding_mlp = Embedding(
            self.num_users + 1, 
            self.latent_dim * 2, 
            embeddings_regularizer=l2(1e-6),
            name='user-embedding-mlp'
        )(user_input)
        user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)
        
        # MLP layers with improved architecture
        concat = Concatenate(name='concat')([product_vec_mlp, user_vec_mlp])
        concat_dropout = Dropout(0.3)(concat)
        fc_1 = Dense(64, name='fc-1', activation='relu', kernel_regularizer=l2(1e-6))(concat_dropout)
        fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
        fc_1_dropout = Dropout(0.3)(fc_1_bn)
        fc_2 = Dense(32, name='fc-2', activation='relu', kernel_regularizer=l2(1e-6))(fc_1_dropout)
        fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
        fc_2_dropout = Dropout(0.2)(fc_2_bn)
        pred_mlp = Dense(16, name='pred-mlp', activation='relu')(fc_2_dropout)
        
        # ===== MF Path =====
        # MF Embeddings with regularization
        product_embedding_mf = Embedding(
            self.num_products + 1, 
            self.latent_dim, 
            embeddings_regularizer=l2(1e-6),
            name='product-embedding-mf'
        )(product_input)
        product_vec_mf = Flatten(name='flatten-product-mf')(product_embedding_mf)
        
        user_embedding_mf = Embedding(
            self.num_users + 1, 
            self.latent_dim, 
            embeddings_regularizer=l2(1e-6),
            name='user-embedding-mf'
        )(user_input)
        user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)
        
        # MF path using element-wise multiply (better than dot for combining)
        pred_mf = Multiply(name='pred-mf-multiply')([product_vec_mf, user_vec_mf])
        
        # ===== Combine MLP and MF =====
        combine_mlp_mf = Concatenate(name='combine-mlp-mf')([pred_mf, pred_mlp])
        
        # Hidden layer before output
        hidden = Dense(16, activation='relu', name='hidden')(combine_mlp_mf)
        
        # Final prediction - output between 1 and 5 (rating scale)
        # Using sigmoid * 4 + 1 to constrain output to [1, 5]
        result_raw = Dense(1, name='result-raw', activation='sigmoid')(hidden)
        result = Lambda(lambda x: x * 4 + 1, name='result')(result_raw)
        
        # Compile model with lower learning rate
        self.model = Model([user_input, product_input], result)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # Lower learning rate
            loss='mse',  # MSE works better for rating prediction
            metrics=['mae']  # Track MAE as metric
        )
        
        print("NeuMF model built successfully")
        self.model.summary()
        return self
        
    def train(self, train_data, epochs=10, batch_size=32, validation_split=0.1):
        """
        Train the model with callbacks for better convergence.
        
        Args:
            train_data (DataFrame): Training data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data for validation
        """
        print(f"Training NeuMF model for {epochs} epochs...")
        
        # Callbacks for better training
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        
        self.history = self.model.fit(
            [train_data.user_id_numeric, train_data.product_id_numeric],
            train_data.rating,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return self.history
        
    def predict(self, user_ids, product_ids, round_decimals=2):
        """
        Make predictions.
        
        Args:
            user_ids: User IDs for prediction
            product_ids: Product IDs for prediction
            round_decimals (int): Number of decimals to round predictions
            
        Returns:
            Predicted ratings
        """
        predictions = self.model.predict([user_ids, product_ids])
        return np.round(predictions, decimals=round_decimals)
    
    def save_model(self, filepath):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """Load a saved model from disk."""
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")
