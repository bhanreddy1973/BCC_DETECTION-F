import os
import json
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'models/checkpoints',
        'data/features',
        'docs/images'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def save_model_data(model, history, X_test, y_test, feature_importance):
    """Save all model data and metrics"""
    try:
        # Save model (already exists, just copying to ensure consistency)
        model.save('models/checkpoints/best_model.h5')
        print("Model saved to models/checkpoints/best_model.h5")

        # Save training history
        with open('logs/training_history.json', 'w') as f:
            json.dump(history, f)
        print("Saved training history to logs/training_history.json")

        # Load existing metrics
        with open('models/checkpoints/test_metrics.json', 'r') as f:
            metrics = json.load(f)
        print("Loaded existing metrics from test_metrics.json")
        
        # Save metrics to logs for visualization
        with open('logs/model_metrics.json', 'w') as f:
            json.dump(metrics, f)
        print("Saved metrics to logs/model_metrics.json")

        # Save feature importance
        joblib.dump(feature_importance, 'models/feature_importance.pkl')
        print("Saved feature importance to models/feature_importance.pkl")

        # Save test data
        np.save('data/features/X_test.npy', X_test)
        np.save('data/features/y_test.npy', y_test)
        print("Saved test data to data/features/")

        print("\nAll data saved successfully!")
        return True

    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def main():
    """Main function to save model data"""
    # Create directories
    create_directories()
    
    try:
        # Load existing model
        model = tf.keras.models.load_model('models/checkpoints/best_model.h5')
        print("Loaded model successfully")
        
        # Load existing training history
        with open('models/checkpoints/training_history.json', 'r') as f:
            history = json.load(f)
        print("Loaded training history successfully")
        
        # Load existing test metrics
        with open('models/checkpoints/test_metrics.json', 'r') as f:
            metrics = json.load(f)
        print("Loaded test metrics successfully")
        
        # Load test data (you'll need to provide this)
        try:
            X_test = np.load('data/features/X_test.npy')
            y_test = np.load('data/features/y_test.npy')
            print("Loaded test data successfully")
        except FileNotFoundError:
            print("Warning: Test data not found. You'll need to provide X_test and y_test.")
            return
        
        # Create feature importance (you'll need to compute this)
        try:
            feature_importance = joblib.load('models/feature_importance.pkl')
            print("Loaded feature importance successfully")
        except FileNotFoundError:
            print("Warning: Feature importance not found. You'll need to compute it.")
            return
        
        # Save all data
        save_model_data(model, history, X_test, y_test, feature_importance)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {str(e)}")
        print("Please ensure all required files exist before running this script.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 