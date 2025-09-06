"""
SpO2 Estimation Module
Implements traditional R-ratio method and ML-based SpO2 estimation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class SpO2Estimator:
    """Main class for SpO2 estimation using traditional and ML methods"""
    
    def __init__(self, sampling_rate=125):
        self.fs = sampling_rate
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        
    def calculate_r_ratio(self, red_ppg, ir_ppg, window_size=None):
        """
        Calculate R-ratio from dual wavelength PPG signals
        
        Args:
            red_ppg: Red wavelength PPG signal (~660nm)
            ir_ppg: Infrared wavelength PPG signal (~940nm)
            window_size: Window size for AC/DC calculation
            
        Returns:
            r_ratio: Calculated R-ratio values
        """
        if window_size is None:
            window_size = int(5 * self.fs)  # 5 second windows
        
        r_ratios = []
        
        for i in range(0, min(len(red_ppg), len(ir_ppg)) - window_size, window_size // 2):
            red_segment = red_ppg[i:i + window_size]
            ir_segment = ir_ppg[i:i + window_size]
            
            # Calculate AC and DC components
            red_ac = np.std(red_segment)  # AC component (variability)
            red_dc = np.mean(red_segment)  # DC component (baseline)
            
            ir_ac = np.std(ir_segment)
            ir_dc = np.mean(ir_segment)
            
            # Calculate R-ratio
            if ir_dc != 0 and red_dc != 0 and ir_ac != 0:
                r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
                r_ratios.append(r_ratio)
            else:
                r_ratios.append(np.nan)
        
        return np.array(r_ratios)
    
    def traditional_spo2_estimation(self, r_ratio):
        """
        Estimate SpO2 using traditional empirical calibration curve
        
        Args:
            r_ratio: R-ratio values
            
        Returns:
            spo2: Estimated SpO2 values
        """
        # Traditional empirical formula (Masimo-style calibration)
        # SpO2 = 110 - 25 * R
        # This is a simplified version; commercial devices use more complex curves
        
        spo2_values = []
        for r in r_ratio:
            if not np.isnan(r) and r > 0:
                # Clamp R-ratio to reasonable range
                r_clamped = np.clip(r, 0.4, 3.0)
                
                # Multiple calibration curves for different ranges
                if r_clamped <= 1.0:
                    # Normal range calibration
                    spo2 = 110 - 25 * r_clamped
                elif r_clamped <= 2.0:
                    # Extended range calibration
                    spo2 = 100 - 15 * (r_clamped - 1.0)
                else:
                    # Low SpO2 range
                    spo2 = 85 - 10 * (r_clamped - 2.0)
                
                # Clamp SpO2 to physiological range
                spo2 = np.clip(spo2, 70, 100)
                spo2_values.append(spo2)
            else:
                spo2_values.append(np.nan)
        
        return np.array(spo2_values)
    
    def prepare_ml_features(self, features_dict, r_ratio=None):
        """
        Prepare features for ML model training
        
        Args:
            features_dict: Dictionary of extracted PPG features
            r_ratio: Optional R-ratio values to include
            
        Returns:
            feature_matrix: 2D array of features
        """
        from ppg_feature_extraction import create_feature_vector
        
        # Extract base features
        feature_vector, feature_names = create_feature_vector(features_dict)
        
        # Add R-ratio if available
        if r_ratio is not None and not np.isnan(r_ratio):
            feature_vector = np.append(feature_vector, r_ratio)
            feature_names.append('r_ratio')
        
        return feature_vector.reshape(1, -1), feature_names
    
    def train_ml_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train multiple ML models for SpO2 estimation
        
        Args:
            X_train: Training features
            y_train: Training SpO2 values
            X_val: Validation features
            y_val: Validation SpO2 values
        """
        print("Training ML models for SpO2 estimation...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Define models with hyperparameter grids
        model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 1.0]
                }
            },
            'neural_network': {
                'model': MLPRegressor(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Train and tune each model
        for model_name, config in model_configs.items():
            print(f"Training {model_name}...")
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Store best model
            self.models[model_name] = grid_search.best_estimator_
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_pred = self.models[model_name].predict(X_val_scaled)
                val_mae = mean_absolute_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                print(f"{model_name} - Validation MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
        
        # Create ensemble model
        self.create_ensemble_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        self.is_trained = True
        print("ML model training completed!")
    
    def create_ensemble_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Create ensemble model combining predictions from multiple models
        
        Args:
            X_train: Training features (scaled)
            y_train: Training SpO2 values
            X_val: Validation features (scaled)
            y_val: Validation SpO2 values
        """
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble")
            return
        
        # Get predictions from all models
        train_predictions = np.column_stack([
            model.predict(X_train) for model in self.models.values()
        ])
        
        # Simple ensemble using weighted average
        # Weights based on validation performance
        if X_val is not None and y_val is not None:
            val_predictions = np.column_stack([
                model.predict(X_val) for model in self.models.values()
            ])
            
            # Calculate weights based on validation MAE (inverse weighting)
            weights = []
            for i, model_name in enumerate(self.models.keys()):
                val_mae = mean_absolute_error(y_val, val_predictions[:, i])
                weights.append(1.0 / (val_mae + 0.01))  # Add small epsilon to avoid division by zero
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
        else:
            # Equal weights if no validation data
            weights = np.ones(len(self.models)) / len(self.models)
        
        self.ensemble_weights = weights
        
        print(f"Ensemble weights: {dict(zip(self.models.keys(), weights))}")
    
    def predict_spo2(self, features_dict, r_ratio=None, method='ensemble'):
        """
        Predict SpO2 using trained models
        
        Args:
            features_dict: Dictionary of extracted PPG features
            r_ratio: Optional R-ratio value
            method: Prediction method ('traditional', 'ml', 'ensemble')
            
        Returns:
            spo2_prediction: Predicted SpO2 value
            confidence: Confidence measure
        """
        if method == 'traditional' and r_ratio is not None:
            spo2_pred = self.traditional_spo2_estimation(np.array([r_ratio]))[0]
            return spo2_pred, 0.8  # Fixed confidence for traditional method
        
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_ml_models first.")
        
        # Prepare features
        feature_vector, feature_names = self.prepare_ml_features(features_dict, r_ratio)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        if method == 'ensemble':
            # Ensemble prediction
            predictions = np.array([
                model.predict(feature_vector_scaled)[0] 
                for model in self.models.values()
            ])
            
            spo2_pred = np.average(predictions, weights=self.ensemble_weights)
            
            # Confidence based on prediction variance
            prediction_std = np.std(predictions)
            confidence = max(0.1, 1.0 - prediction_std / 10.0)  # Higher std = lower confidence
            
        else:
            # Single model prediction
            if method not in self.models:
                method = list(self.models.keys())[0]  # Default to first model
            
            spo2_pred = self.models[method].predict(feature_vector_scaled)[0]
            confidence = 0.7  # Fixed confidence for single model
        
        # Clamp to physiological range
        spo2_pred = np.clip(spo2_pred, 70, 100)
        
        return spo2_pred, confidence
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate trained models on test data
        
        Args:
            X_test: Test features
            y_test: True SpO2 values
            
        Returns:
            evaluation_results: Dictionary of performance metrics
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_ml_models first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            
            results[model_name] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'predictions': y_pred
            }
        
        # Evaluate ensemble
        ensemble_predictions = np.column_stack([
            model.predict(X_test_scaled) for model in self.models.values()
        ])
        ensemble_pred = np.average(ensemble_predictions, weights=self.ensemble_weights, axis=1)
        
        results['ensemble'] = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'r2': r2_score(y_test, ensemble_pred),
            'predictions': ensemble_pred
        }
        
        return results
    
    def save_models(self, filepath):
        """Save trained models and scaler"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'ensemble_weights': self.ensemble_weights if hasattr(self, 'ensemble_weights') else None,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load trained models and scaler"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.ensemble_weights = model_data.get('ensemble_weights', None)
        self.is_trained = model_data.get('is_trained', True)
        print(f"Models loaded from {filepath}")

# Utility functions
def generate_synthetic_training_data(n_samples=1000):
    """
    Generate synthetic training data for SpO2 estimation
    This would be replaced with real dataset loading in practice
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Feature matrix
        y: SpO2 values
    """
    np.random.seed(42)
    
    # Generate synthetic features that correlate with SpO2
    n_features = 25
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic SpO2 values based on features
    # This is a simplified simulation - real relationships would be more complex
    base_spo2 = 98
    
    # Heart rate effect (higher HR might indicate lower SpO2)
    hr_effect = -0.05 * X[:, 0]  # Assuming first feature is HR-related
    
    # Signal quality effect
    quality_effect = 2 * X[:, 1]  # Assuming second feature is quality-related
    
    # R-ratio effect (most important for SpO2)
    r_ratio_effect = -8 * X[:, -1]  # Assuming last feature is R-ratio
    
    # Random individual variation
    individual_variation = np.random.normal(0, 1.5, n_samples)
    
    y = base_spo2 + hr_effect + quality_effect + r_ratio_effect + individual_variation
    
    # Add some realistic SpO2 distribution
    low_spo2_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[low_spo2_indices] = np.random.uniform(85, 95, len(low_spo2_indices))
    
    # Clamp to physiological range
    y = np.clip(y, 70, 100)
    
    return X, y

# Example usage and testing
if __name__ == "__main__":
    print("Testing SpO2 Estimation Pipeline...")
    
    # Generate synthetic training data
    X_synthetic, y_synthetic = generate_synthetic_training_data(1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Initialize SpO2 estimator
    spo2_estimator = SpO2Estimator(sampling_rate=125)
    
    # Train models
    spo2_estimator.train_ml_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    results = spo2_estimator.evaluate_model(X_test, y_test)
    
    print("\nModel Performance on Test Set:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
    
    # Test traditional method
    r_ratio_example = 0.8
    traditional_spo2 = spo2_estimator.traditional_spo2_estimation(np.array([r_ratio_example]))[0]
    print(f"\nTraditional SpO2 estimation (R={r_ratio_example}): {traditional_spo2:.1f}%")
    
    print("SpO2 estimation pipeline completed successfully!")