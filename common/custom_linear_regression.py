import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error

class CustomLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coefficients_ = None
        
    def _fit_base(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_matrix = np.c_[np.ones(shape=(X.shape[0],1)), X.values]
        else: 
            X_matrix = np.c_[np.ones(shape=(X.shape[0],1)), X]
            
        if isinstance(y, pd.Series):
            y_matrix = y.values.reshape(-1, 1)
        else: 
            y_matrix = y.reshape(-1, 1)
            
        return X_matrix, y_matrix
    
    def predict(self, X):
        if self.coefficients_ is None:
            raise ValueError("Model is not trained.")
        
        if isinstance(X, pd.DataFrame):
            return np.c_[np.ones((X.shape[0], 1)), X.values] @ self.coefficients_
        
        return np.c_[np.ones((X.shape[0], 1)), X] @ self.coefficients_
    
    def transform(self, X):
        return self.predict(X)
    
    
class LinearRegressionClosedForm(CustomLinearRegression):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y, should_scale=False):
        X_matrix, y_matrix = self._fit_base(X, y)
        
        if should_scale:
            X_matrix = StandardScaler().fit(X_matrix)
        
        self.coefficients_ = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_matrix
        return self
        

class LinearRegressionGradientDescent(CustomLinearRegression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y, **kwargs):
        epochs = kwargs.get("epochs", 500)
        batch_size = kwargs.get("batch_size", None)
        learning_rate = kwargs.get("learning_rate", 0.01)
        should_scale = kwargs.get("should_scale", False)
        mini_batch = kwargs.get("mini_batch", True)
        
        X_matrix, y_matrix = self._fit_base(X, y)
        
        if should_scale:
            X_matrix = StandardScaler().fit(X_matrix)
            
        samples_count, feat_count = X_matrix.shape
        self.coefficients_ = np.zeros((feat_count, 1))
        
        for _ in range(epochs):
            
            if mini_batch and batch_size:           # mini batch gradient descent
                indices = np.random.permutation(samples_count)
                for i in range(0, samples_count, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_matrix[batch_indices]
                    y_batch = y_matrix[batch_indices]
                    
                    gradients = (2 / len(X_batch)) * X_batch.T @ (X_batch @ self.coefficients_ - y_batch)
                    self.coefficients_ -= learning_rate * gradients
                        
            else:
                
                if batch_size:                      # stochastic batch gradient descent
                    indices = np.random.choice(samples_count, batch_size, replace = False)
                    X_batch = X_matrix[indices]
                    y_batch = y_matrix[indices]
                    factor = 2 / batch_size
                else:                               # simple batch gradient descent
                    X_batch = X_matrix
                    y_batch = y_matrix
                    factor = 2 / samples_count
                    
                gradients = factor * X_batch.T @ (X_batch @ self.coefficients_ - y_batch)
                self.coefficients_ -= learning_rate * gradients
        
        return self
            
        
        
        
PARAM_CANDIDATES = {
    "epochs": [2500, 5000, 7500, 10000],
    "batch_sizes": [None, 64, 128, 256],
    "learning_rates": [0.0005, 0.001, 0.005]
}


def find_gradient_descent_best_params(X_train, y_train, X_val, y_val, should_log=False):
    
    model = LinearRegressionGradientDescent()
    best_mse = float('inf')
    best_params = None
    
    for epochs in PARAM_CANDIDATES["epochs"]:
        for batch_size in PARAM_CANDIDATES["batch_sizes"]:
            for learning_rate in PARAM_CANDIDATES["learning_rates"]:
                kwargs = {
                    "epochs": epochs, 
                    "batch_size": batch_size, 
                    "learning_rate": learning_rate
                }
                model.fit(X_train, y_train, **kwargs)
                mse = mean_squared_error(y_val, model.predict(X_val))
                if mse < best_mse:
                    best_mse = mse
                    best_params = kwargs.copy()
                
        # log (but it's not log XD)    
        if should_log:
            print("\tepochs = ", epochs)
                    
    return best_params
    
    
    

    
