import os
import sys
import numpy as np
import pandas as pd
from src.exceptions import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import dill

def get_cabin_desk(cabin):
    if type(cabin)  == str:
        return cabin[0]
    else:
        return None
    
def get_cabin_side(cabin):    
    if type(cabin)  == str:
        return cabin[-1]
    else:
        return None


def load_object(file_path):
    try: 
        with open(file_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train , X_test, y_test, models, params):
    try:
        report = {}
        for name_model, model in models.items():
            model_params = params[name_model]

            gs = GridSearchCV(model, model_params, cv=4)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred  = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[name_model] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)