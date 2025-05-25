from common.consts import DATA_PATH, CATEGORICAL_COLUMN_NAMES, CATEGORY_TRANSLATIONS
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():  
    data = pd.read_csv(DATA_PATH, sep=";")
    data.columns = data.columns.str.strip()
    
    for column_name in CATEGORICAL_COLUMN_NAMES:
        data[column_name] = data[column_name].map(lambda field: CATEGORY_TRANSLATIONS.get(column_name, {}).get(field, field))
        
    return data

def get_numeric_data():
    data = pd.read_csv(DATA_PATH, sep=";")
    data.columns = data.columns.str.strip()
    data_numeric = data.drop(columns=CATEGORICAL_COLUMN_NAMES)
    
    return data_numeric
    
def split(X, y):
        
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=30)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.6, random_state=30)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    