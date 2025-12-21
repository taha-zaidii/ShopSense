"""
Flask web application for the Product Recommendation System.
"""
import os
import sys
import pickle

# Force TensorFlow to run eagerly (fixes TensorShape error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.run_functions_eagerly(True)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from models import MatrixFactorizationModel, NeuMFModel
from evaluator import ModelEvaluator
from recommender import ProductRecommender
from verification import RecommendationVerifier
import config

app = Flask(__name__)
CORS(app)

# Global variables to store models and data
data_loader = None
model = None
recommender = None
training_status = {"status": "not_started", "progress": 0, "message": ""}
model_metrics = {}


def initialize_or_load_model():
    """Initialize or load pre-trained model."""
    global data_loader, model, recommender, model_metrics
    
    model_path = os.path.join(config.MODELS_DIR, 'neumf_model.keras')
    data_path = os.path.join(config.MODELS_DIR, 'data_loader.pkl')
    metrics_path = os.path.join(config.MODELS_DIR, 'metrics.pkl')
    
    # Check if saved model exists
    if os.path.exists(model_path) and os.path.exists(data_path):
        print("Loading pre-trained model...")
        
        # Load data loader
        with open(data_path, 'rb') as f:
            data_loader = pickle.load(f)
        
        # Initialize and load model
        model = NeuMFModel(
            num_users=data_loader.num_users,
            num_products=data_loader.num_products,
            latent_dim=config.LATENT_DIM
        )
        model.load_model(model_path)
        
        # Load metrics if available
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                model_metrics = pickle.load(f)
        
        # Create recommender
        recommender = ProductRecommender(model, data_loader)
        
        print("Model loaded successfully!")
        return True
    
    return False


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status."""
    global data_loader, model, training_status, model_metrics
    
    return jsonify({
        'model_loaded': model is not None,
        'data_loaded': data_loader is not None,
        'training_status': training_status,
        'metrics': model_metrics,
        'num_users': data_loader.num_users if data_loader else 0,
        'num_products': data_loader.num_products if data_loader else 0,
        'dataset_size': len(data_loader.dataset) if data_loader else 0
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the recommendation model."""
    global data_loader, model, recommender, training_status, model_metrics
    
    try:
        data = request.json
        sample_size = data.get('sample_size', 10000)
        epochs = data.get('epochs', 5)
        
        training_status = {"status": "training", "progress": 10, "message": "Loading data..."}
        
        # Load data
        data_file_path = os.path.join(config.DATA_DIR, config.DATA_FILE)
        data_loader = DataLoader(data_file_path, sample_size=sample_size)
        data_loader.load_data().preprocess_data()
        
        training_status["progress"] = 30
        training_status["message"] = "Splitting data..."
        
        # Split data
        train_data, test_data = data_loader.split_data(
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        training_status["progress"] = 40
        training_status["message"] = "Building model..."
        
        # Build and train NeuMF model
        model = NeuMFModel(
            num_users=data_loader.num_users,
            num_products=data_loader.num_products,
            latent_dim=config.LATENT_DIM
        )
        model.build_model()
        
        training_status["progress"] = 50
        training_status["message"] = f"Training for {epochs} epochs..."
        
        # Train model
        history = model.train(train_data, epochs=epochs, batch_size=config.BATCH_SIZE)
        
        training_status["progress"] = 80
        training_status["message"] = "Evaluating model..."
        
        # Evaluate model
        metrics = ModelEvaluator.evaluate_model(model, test_data)
        model_metrics = metrics
        
        training_status["progress"] = 90
        training_status["message"] = "Saving model..."
        
        # Save model and data loader
        model_path = os.path.join(config.MODELS_DIR, 'neumf_model.keras')
        data_path = os.path.join(config.MODELS_DIR, 'data_loader.pkl')
        metrics_path = os.path.join(config.MODELS_DIR, 'metrics.pkl')
        
        model.save_model(model_path)
        with open(data_path, 'wb') as f:
            pickle.dump(data_loader, f)
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Create recommender
        recommender = ProductRecommender(model, data_loader)
        
        training_status = {"status": "completed", "progress": 100, "message": "Training completed!"}
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'num_users': data_loader.num_users,
            'num_products': data_loader.num_products
        })
        
    except Exception as e:
        training_status = {"status": "error", "progress": 0, "message": str(e)}
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get product recommendations for a user."""
    global recommender, data_loader
    
    if not recommender or not data_loader:
        return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
    
    try:
        data = request.json
        user_id = data.get('user_id')
        top_n = data.get('top_n', 10)
        
        # Validate user_id
        if user_id is None:
            # Get a random user
            user_id = int(data_loader.dataset.user_id_numeric.iloc[0])
        else:
            user_id = int(user_id)
            
        # Check if user exists
        unique_users = data_loader.dataset.user_id_numeric.unique()
        if user_id not in unique_users:
            # Use the first available user
            user_id = int(unique_users[0])
        
        # Get recommendations
        recommendations = recommender.recommend_products(user_id, top_n=top_n)
        
        # Format recommendations
        formatted_recs = [
            {
                'product_id': int(prod_id),
                'predicted_rating': float(rating)
            }
            for prod_id, rating in recommendations
        ]
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'recommendations': formatted_recs
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/random_user')
def get_random_user():
    """Get a random user ID."""
    global data_loader
    
    if not data_loader:
        return jsonify({'success': False, 'error': 'Data not loaded'}), 400
    
    import random
    users = data_loader.dataset.user_id_numeric.unique()
    random_user = int(random.choice(users))
    
    return jsonify({
        'success': True,
        'user_id': random_user
    })


@app.route('/api/verify', methods=['POST'])
def verify_recommendations():
    """Verify recommendation quality with ranking metrics."""
    global model, data_loader
    
    if not model or not data_loader:
        return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
    
    try:
        data = request.json
        sample_users = data.get('sample_users', 30)
        threshold = data.get('threshold', 4.0)
        
        # Create verifier
        verifier = RecommendationVerifier(model, data_loader)
        
        # Run ranking metrics evaluation
        ranking_metrics = verifier.evaluate_all_users(
            sample_users=sample_users,
            k_values=[5, 10, 20],
            threshold=threshold,
            verbose=False
        )
        
        # Run leave-one-out evaluation
        loo_metrics = verifier.leave_one_out_evaluation(
            sample_users=sample_users,
            k_values=[5, 10, 20],
            verbose=False
        )
        
        # Combine metrics
        all_metrics = {**ranking_metrics, **loo_metrics}
        
        # Format for frontend
        formatted_metrics = {}
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                formatted_metrics[key] = round(value, 4)
            else:
                formatted_metrics[key] = value
        
        return jsonify({
            'success': True,
            'metrics': formatted_metrics,
            'interpretation': {
                'precision': 'What % of recommended items are actually relevant',
                'recall': 'What % of relevant items appear in recommendations',
                'hit_rate': 'Does at least 1 relevant item appear in top K',
                'ndcg': 'Are relevant items ranked higher (considers position)',
                'loo_hit': 'Can model predict users highest-rated item'
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/compare_user', methods=['POST'])
def compare_user_recommendations():
    """Compare recommendations with a user's actual ratings."""
    global model, data_loader
    
    if not model or not data_loader:
        return jsonify({'success': False, 'error': 'Model not trained yet'}), 400
    
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if user_id is None:
            # Get a random user
            import random
            users = data_loader.dataset.user_id_numeric.unique()
            user_id = int(random.choice(users))
        else:
            user_id = int(user_id)
        
        # Create verifier
        verifier = RecommendationVerifier(model, data_loader)
        
        # Get comparison
        comparison = verifier.compare_with_actual_ratings(user_id, top_n=10)
        
        if 'error' in comparison:
            return jsonify({'success': False, 'error': comparison['error']}), 400
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def find_free_port(start_port=5000, max_attempts=10):
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")


if __name__ == '__main__':
    print("="*60)
    print("Product Recommendation System - Web Interface")
    print("="*60)
    
    # Try to load pre-trained model (non-blocking)
    # if initialize_or_load_model():
    #     print("\n✓ Pre-trained model loaded successfully!")
    #     print(f"✓ Users: {data_loader.num_users}, Products: {data_loader.num_products}")
    # else:
    print("\n⚠ No pre-trained model found.")
    print("Please train the model using the web interface.")
    
    # Find a free port automatically
    port = find_free_port(start_port=5000)
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print(f"Access the web interface at: http://localhost:{port}")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)

