from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from common.consts import DATA_PATH, CATEGORICAL_COLUMN_NAMES, CATEGORY_TRANSLATIONS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_data(only_numeric=False):  
    data = pd.read_csv(DATA_PATH, sep=";")
    data.columns = data.columns.str.strip()
    
    if only_numeric:
        data = data.drop(columns=CATEGORICAL_COLUMN_NAMES)
    else:
        for column_name in CATEGORICAL_COLUMN_NAMES:
            data[column_name] = data[column_name].map(lambda field: CATEGORY_TRANSLATIONS.get(column_name, {}).get(field, field))
        
        
    return data
    
def split(X, y):
        
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=30)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.6, random_state=30)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_accurancy_dataframe(model, preprocessor, X_train_val, y_train_val, X_test, y_test, cv=5):    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    
    results = np.empty((0, 3))
    
    for _ in range(cv):
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2)
        
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_val = pipeline.predict(X_val)
        y_pred_test = pipeline.predict(X_test)
        
        results = np.vstack([results, np.array([
            accuracy_score(y_train, y_pred_train), 
            accuracy_score(y_val, y_pred_val),
            accuracy_score(y_test, y_pred_test)
        ])])
        
    result_df = pd.DataFrame(results.T, index=["Train", "Validation", "Test"])
    return result_df

    