"""
Configuration settings for the recommendation system.
"""
import os

# Data configuration
DATA_FILE = 'ratings_Electronics (1).csv'
SAMPLE_SIZE = 50000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model configuration
LATENT_DIM = 10
EPOCHS = 10
BATCH_SIZE = 32

# Model types
MODEL_TYPE_MF = 'matrix_factorization'
MODEL_TYPE_NEUMF = 'neumf'

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
