import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src.exceptions import CustomException
from src.utils import get_cabin_desk, get_cabin_side
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,cat_cols, num_cols ):
        """
        This function is responsible for data transformation 
        """
        try:

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("lg", FunctionTransformer(np.log1p)),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns : {num_cols}")
            logging.info(f"Categorical columns : {cat_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read Train and test data completed")
            
            train_df['cabin_desk'] = train_df['Cabin'].apply(get_cabin_desk)
            train_df['cabin_side'] = train_df['Cabin'].apply(get_cabin_side) #[get_cabin_desk(c) if type(c) == str else None for c in train_df['Cabin'] ]
            test_df['cabin_desk'] = test_df['Cabin'].apply(get_cabin_desk)
            test_df['cabin_side'] = test_df['Cabin'].apply(get_cabin_side)

            cols_to_drop = ['PassengerId', 'Cabin', "Name"]
            train_df.drop(columns=cols_to_drop, inplace=True)
            test_df.drop(columns=cols_to_drop, inplace=True)

            logging.info("Feature Engineering done")

            logging.info("Obtaining preprocessing object")
            
            target_column_name = 'Transported'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            cat_cols = input_feature_train_df.select_dtypes(include='O').columns.tolist()
            num_cols = input_feature_train_df.select_dtypes(exclude='O').columns.tolist()

            preprocessing_obj = self.get_data_transformer_object(cat_cols, num_cols)

            logging.info("Applying preprocessing object on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing object saved")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)