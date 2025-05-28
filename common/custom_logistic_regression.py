import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, RegressorMixin


class CustomLogisticRegressionBinary(BaseEstimator, RegressorMixin):
   
    def __init__(self):
        self.coefficients_ = None
        
    def predict_proba(self, X):
        if self.coefficients_ is None:
            raise ValueError("Model is not trained.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if scipy.sparse.issparse(X):
            X = X.toarray()   
            
        X_matrix = np.c_[np.ones((X.shape[0], 1)), X]
        return 1 / (1 + np.exp(- X_matrix @ self.coefficients_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def _fit_base(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values  
        if scipy.sparse.issparse(X):
            X = X.toarray()    
        
        X_matrix = np.c_[np.ones(shape=(X.shape[0],1)), X]
            
        if isinstance(y, pd.Series):
            y = y.values
        y_matrix = y.reshape(-1, 1)
            
        return X_matrix, y_matrix
    
    
    def fit(self, X, y, **kwargs):
        epochs = kwargs.get("epochs", 500)
        batch_size = kwargs.get("batch_size", None)
        learning_rate = kwargs.get("learning_rate", 0.01)
        mini_batch = kwargs.get("mini_batch", True)
        
        X_matrix, y_matrix = self._fit_base(X, y)
        
        samples_count, feat_count = X_matrix.shape
        self.coefficients_ = np.zeros((feat_count, 1))
        
        for _ in range(epochs):
            
            if mini_batch and batch_size:           # mini batch gradient descent
                indices = np.random.permutation(samples_count)
                for i in range(0, samples_count, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_matrix[batch_indices]
                    y_batch = y_matrix[batch_indices]
                    
                    sigmoid = 1 / (1 + np.exp(-X_batch @ self.coefficients_))
                    gradients = X_batch.T @ (sigmoid - y_batch) / len(X_batch)
                    self.coefficients_ -= learning_rate * gradients
            else:
                
                if batch_size:                      # stochastic batch gradient descent
                    indices = np.random.choice(samples_count, batch_size, replace = False)
                    X_batch = X_matrix[indices]
                    y_batch = y_matrix[indices]
                    factor = 1 / batch_size
                else:                               # simple batch gradient descent
                    X_batch = X_matrix
                    y_batch = y_matrix
                    factor = 1 / samples_count
                    
                sigmoid = 1 / (1 + np.exp(-X_batch @ self.coefficients_))
                gradients = factor * X_batch.T @ (sigmoid - y_batch)
                self.coefficients_ -= learning_rate * gradients

    def transform(self, X):
            return self.predict(X)
    
    


class CustomLogisticRegressionMulticlass(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.binary_models = {}
        self.class_labels = None
        
      
    def fit(self, X, y, **kwargs):
        self.class_labels = np.unique(y)
        for class_label in self.class_labels:
            y_bin = (y == class_label).astype(int)
            binary_model = CustomLogisticRegressionBinary()
            binary_model.fit(X, y_bin, **kwargs)
            self.binary_models[class_label] = binary_model
        return self
    
    def predict_proba(self, X):
        return np.column_stack([
            self.binary_models[class_label].predict_proba(X).ravel() 
            for class_label in self.class_labels
        ])
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def transform(self, X):
        return self.predict(X)
        
            
            
            
            
            
            