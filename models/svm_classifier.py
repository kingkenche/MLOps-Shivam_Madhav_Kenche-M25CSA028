"""
SVM Classifier for MNIST and FashionMNIST
Question 1(b)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time
import pickle
import os


class SVMClassifier:
    """SVM Classifier wrapper with timing and evaluation"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', max_iter=-1):
        """
        Initialize SVM classifier
        
        Args:
            kernel: 'rbf', 'poly', 'linear', 'sigmoid'
            C: Regularization parameter
            gamma: Kernel coefficient
            max_iter: Maximum iterations (-1 for no limit)
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, max_iter=max_iter, 
                        cache_size=1000, verbose=False)
        self.train_time = 0
        self.test_time = 0
        
    def train(self, X_train, y_train):
        """Train the SVM model"""
        print(f"\nTraining SVM with kernel='{self.kernel}', C={self.C}, gamma='{self.gamma}'")
        print(f"Training samples: {X_train.shape[0]}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"Training completed in {self.train_time:.2f} ms")
        return self.train_time
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        start_time = time.time()
        y_pred = self.predict(X_test)
        self.test_time = (time.time() - start_time) * 1000  # Convert to ms
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print(f"Test Time: {self.test_time:.2f} ms")
        
        return accuracy, self.test_time
    
    def save_model(self, filepath):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


def train_svm_variants(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name='MNIST', save_dir='./results'):
    """
    Train SVM with different kernel types and hyperparameters
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        dataset_name: Name of the dataset (MNIST or FashionMNIST)
        save_dir: Directory to save trained models
    
    Returns:
        Dictionary with results for each configuration
    """
    results = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define hyperparameter configurations
    configs = [
        # RBF kernel with different C and gamma values
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        
        # Polynomial kernel with different degrees
        {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'degree': 2},
        {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'degree': 3},
        {'kernel': 'poly', 'C': 10.0, 'gamma': 'scale', 'degree': 2},
    ]
    
    print(f"\n{'='*60}")
    print(f"Training SVM variants on {dataset_name}")
    print(f"{'='*60}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Configuration {i}/{len(configs)} ---")
        print(f"Config: {config}")
        
        # Create and train SVM
        if 'degree' in config:
            degree = config.pop('degree')
            svm = SVC(**config, degree=degree, max_iter=10000, cache_size=1000)
            config['degree'] = degree  # Add it back for results
        else:
            svm = SVC(**config, max_iter=10000, cache_size=1000)
        
        # Train
        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time_ms = (time.time() - start_time) * 1000
        
        # Evaluate on validation set
        val_acc = accuracy_score(y_val, svm.predict(X_val)) * 100
        
        # Evaluate on test set
        start_time = time.time()
        y_pred = svm.predict(X_test)
        test_time_ms = (time.time() - start_time) * 1000
        test_acc = accuracy_score(y_test, y_pred) * 100
        
        # Create model filename
        kernel = config['kernel']
        C_val = config.get('C', 1.0)
        gamma_val = config.get('gamma', 'scale')
        degree_val = config.get('degree', 'N/A')
        
        if degree_val != 'N/A':
            model_filename = f"{dataset_name.lower()}_svm_{kernel}_C{C_val}_gamma{gamma_val}_deg{degree_val}_best.pth"
        else:
            model_filename = f"{dataset_name.lower()}_svm_{kernel}_C{C_val}_gamma{gamma_val}_best.pth"
        
        model_path = os.path.join(save_dir, model_filename)
        
        # Save the trained model using pickle (stored in .pth file for consistency)
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)
        
        print(f"Model saved to: {model_path}")
        
        result = {
            'config': config.copy(),
            'train_time_ms': train_time_ms,
            'test_time_ms': test_time_ms,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'dataset': dataset_name,
            'model_path': model_path
        }
        results.append(result)
        
        print(f"Training time: {train_time_ms:.2f} ms")
        print(f"Validation accuracy: {val_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        print(f"Test time: {test_time_ms:.2f} ms")
    
    return results


def print_svm_results(results):
    """Print SVM results in a formatted table"""
    print(f"\n{'='*80}")
    print("SVM Results Summary")
    print(f"{'='*80}\n")
    
    print(f"{'Kernel':<8} {'C':<8} {'Gamma':<10} {'Degree':<8} {'Val Acc':<10} {'Test Acc':<10} {'Train Time (ms)':<18} {'Test Time (ms)':<15}")
    print("-" * 80)
    
    for result in results:
        config = result['config']
        kernel = config['kernel']
        C = config.get('C', 'N/A')
        gamma = config.get('gamma', 'N/A')
        degree = config.get('degree', 'N/A')
        val_acc = result['val_accuracy']
        test_acc = result['test_accuracy']
        train_time = result['train_time_ms']
        test_time = result['test_time_ms']
        
        print(f"{kernel:<8} {C:<8} {str(gamma):<10} {str(degree):<8} "
              f"{val_acc:<10.2f} {test_acc:<10.2f} {train_time:<18.2f} {test_time:<15.2f}")


def load_svm_model(model_path):
    """
    Load a saved SVM model from a .pth file
    
    Args:
        model_path: Path to the saved model file
    
    Returns:
        Loaded SVM model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    return model


def evaluate_saved_svm(model_path, X_test, y_test):
    """
    Load a saved SVM model and evaluate it on test data
    
    Args:
        model_path: Path to the saved model file
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with test accuracy and inference time
    """
    # Load model
    svm = load_svm_model(model_path)
    
    # Evaluate
    start_time = time.time()
    y_pred = svm.predict(X_test)
    test_time_ms = (time.time() - start_time) * 1000
    
    test_acc = accuracy_score(y_test, y_pred) * 100
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Time: {test_time_ms:.2f} ms")
    
    return {
        'test_accuracy': test_acc,
        'test_time_ms': test_time_ms,
        'predictions': y_pred
    }


if __name__ == "__main__":
    print("SVM classifier module loaded successfully!")
    print("Example usage:")
    print("  from models.svm_classifier import SVMClassifier")
    print("  svm = SVMClassifier(kernel='rbf', C=1.0)")
    print("  svm.train(X_train, y_train)")
    print("  accuracy, test_time = svm.evaluate(X_test, y_test)")
    print("\nFor batch training:")
    print("  from models.svm_classifier import train_svm_variants")
    print("  results = train_svm_variants(X_train, y_train, X_val, y_val, X_test, y_test)")
    print("\nTo load saved models:")
    print("  from models.svm_classifier import load_svm_model, evaluate_saved_svm")
    print("  results = evaluate_saved_svm('model.pth', X_test, y_test)")
