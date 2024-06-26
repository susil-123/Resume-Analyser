from dataclasses import dataclass
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
logistic_regression = LogisticRegression()
from sklearn.metrics import accuracy_score
from src.utils import load_object,save_obj
from src.logger import logging
from src.exception import CustomException

class ModelTrainingConfig:
    vectorizer_path = os.path.join('artifacts','vectorizer.pkl')
    model_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training = ModelTrainingConfig()
    
    def initiate_model_training(self,resume_df_path):
        try:
            resume_df = load_object(file_path=resume_df_path)

            vector_df = pd.DataFrame(vectorizer.fit_transform(resume_df['Resume']).toarray(),columns=vectorizer.get_feature_names_out())
            logging.info("Resume and Vector loaded from path")

            X = vector_df
            y = resume_df['Category_encoded']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            logistic_regression.fit(X_train,y_train)
            score = accuracy_score(y_test,logistic_regression.predict(X_test))
            print(f"Score of the model: {score}")

            save_obj(file_path=self.model_training.vectorizer_path,obj=vectorizer)
            save_obj(file_path=self.model_training.model_path,obj=logistic_regression)
            logging.info("Resume and Vector are saved as pickle files")

            return (
                self.model_training.vectorizer_path,
                self.model_training.model_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        