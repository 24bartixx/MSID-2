from common.consts import DATA_PATH, CATEGORICAL_COLUMN_NAMES, CATEGORY_TRANSLATIONS
import pandas as pd


def read_data():  
    data = pd.read_csv(DATA_PATH, sep=";")
    data.columns = data.columns.str.strip()
    
    for column_name in CATEGORICAL_COLUMN_NAMES:
        data[column_name] = data[column_name].map(lambda field: CATEGORY_TRANSLATIONS.get(column_name, {}).get(field, field))
    
    return data
    