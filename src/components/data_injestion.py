import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.pipelines.predict_pipeline import PredictPipeline

@dataclass
class DataInjestionConfig:
    resume_data_path = os.path.join('artifacts','resume_df.csv')
    skill_data_path = os.path.join('artifacts','skill_df.csv')

class DataInjestion:
    def __init__(self):
        self.data_injestion_config = DataInjestionConfig()

    def initiate_data_injestion(self):
        try:
            resume_df = pd.read_csv('./data/resume_dataset.csv')
            skill_df = pd.read_csv('./data/related_skills.csv')
            logging.info("Read the datasets")

            os.makedirs(os.path.dirname(self.data_injestion_config.resume_data_path),exist_ok=True)
            logging.info("Directory created")           

            resume_df.to_csv(self.data_injestion_config.resume_data_path)
            skill_df.to_csv(self.data_injestion_config.skill_data_path)
            logging.info("Csv files saved on artifacts")
            
            return (
                self.data_injestion_config.resume_data_path,
                self.data_injestion_config.skill_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_injestion = DataInjestion()
    resume_path,skill_path = data_injestion.initiate_data_injestion()
    data_transformation_obj = DataTransformation()
    resume_pr,skill_pr = data_transformation_obj.initiate_data_transformation(resume_path,skill_path)
    model_training_obj = ModelTraining()
    model_training_obj.initiate_model_training(resume_pr)
    predict_obj = PredictPipeline()
    predict_obj.get_output()
    
            