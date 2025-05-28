import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

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
    
    def fit(self, X, y):
        X_matrix, y_matrix = self._fit_base(X, y)
        self.coefficients_ = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_matrix
        return self
        

class LinearRegressionGradientDescent(CustomLinearRegression):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y, **kwargs):
        epochs = kwargs.get("epochs", 500)
        batch_size = kwargs.get("batch_size", None)
        learning_rate = kwargs.get("learning_rate", 0.01)
        
        X_matrix, y_matrix = self._fit_base(X, y)
        
        samples_count, feat_count = X_matrix.shape
        
        self.coefficients_ = np.zeros((feat_count, 1))
        
        for i in range(epochs):
            if batch_size:
                indices = np.random.choice(samples_count, batch_size, replace = False)
                X_batch = X_matrix[indices]
                y_batch = y_matrix[indices]
            else:           # simple batch gradient descent
                X_batch = X_matrix
                y_batch = y_matrix
                
            factor = (2 / (batch_size if batch_size else samples_count)) 
            gradients = factor * X_batch.T @ (X_batch @ self.coefficients_ - y_batch)
                    
            self.coefficients_ -= learning_rate * gradients
        
        return self
            
        
    
    
    

    
