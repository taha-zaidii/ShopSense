"""
Model evaluation utilities.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ModelEvaluator:
    """Utility class for evaluating recommendation models."""
    
    @staticmethod
    def plot_training_history(history, metric='loss', title='Training History'):
        """
        Plot training history.
        
        Args:
            history: Keras history object
            metric (str): Metric to plot (default: 'loss')
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric])
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            dict: Dictionary containing various metrics
        """
        # Flatten predictions if needed
        y_pred_flat = np.array(y_pred).flatten()
        y_true_flat = np.array(y_true).flatten()
        
        # Error metrics
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        
        # Accuracy metrics
        # Round predictions to nearest integer for accuracy calculation
        y_pred_rounded = np.clip(np.round(y_pred_flat), 1, 5)
        y_true_rounded = np.round(y_true_flat)
        
        # Exact accuracy: predicted rating (rounded) matches actual
        exact_accuracy = np.mean(y_pred_rounded == y_true_rounded) * 100
        
        # Within-1 accuracy: prediction is within 1 star of actual
        within_1_accuracy = np.mean(np.abs(y_pred_flat - y_true_flat) <= 1.0) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Accuracy': exact_accuracy,
            'Within-1 Accuracy': within_1_accuracy
        }
        
        return metrics
        
    @staticmethod
    def print_metrics(metrics):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print("="*50 + "\n")
        
    @staticmethod
    def evaluate_model(model, test_data):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model (MatrixFactorizationModel or NeuMFModel)
            test_data (DataFrame): Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Make predictions
        y_pred = model.predict(
            test_data.user_id_numeric,
            test_data.product_id_numeric
        )
        y_true = test_data.rating.values
        
        # Calculate metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        
        # Print metrics
        ModelEvaluator.print_metrics(metrics)
        
        return metrics
