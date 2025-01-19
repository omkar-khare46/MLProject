import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@ dataclass
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered the model training process")
            logging.info("split the training and test input data")
            X_train,y_train, X_test, y_test = (train_array[:,:-1],
                                              train_array[:,-1],
                                              test_array[:,:-1],
                                              test_array[:,-1]
                                                )
            
            models = {
            'linear' : LinearRegression(), 'gradient boosting': GradientBoostingRegressor(), 'KNN' : KNeighborsRegressor(), 'Decision tree' : DecisionTreeRegressor(), 
            'random forest': RandomForestRegressor(), 'catboost': CatBoostRegressor(verbose=0), 'adaboost': AdaBoostRegressor(), 'xgboost': XGBRegressor()
                }
            logging.info(f"Evaluating models...")
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train, X_test =X_test, 
                                               y_test = y_test, models = models )
            logging.info(f"Model Report: {model_report}")
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model from the dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model =    models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("No best model was found")
            logging.info("Best model found on both training and test dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj = best_model)
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R² Score of the best model on test data: {r2_square}")
            return r2_square

        except Exception as e:
            CustomException(e, sys)