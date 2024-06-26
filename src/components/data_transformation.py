from dataclasses import dataclass
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import sys
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessed_resume_df_path = os.path.join('artifacts','preprocessed_resume_df.pkl')
    preprocessed_skill_df_path = os.path.join('artifacts','preprocessed_skill_df.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def preprocessing(self,text):
        words = word_tokenize(text,language='english')
        words = [lemmatizer.lemmatize(word,pos='v') for word in words if word not in stopwords.words('english')]
        words = [i.replace(',','') for i in words]
        words = [i.replace('.','') for i in words]
        words = [i.replace('*','') for i in words]
        words = [i.replace('-','') for i in words]
        words = [i.replace(':','') for i in words]
        return ' '.join(words)

    def lower_text(self,text):
        return text.lower()

    def initiate_data_transformation(self,resume_df,skill_df):
        try:
            resume_df = pd.read_csv(resume_df)
            skill_df = pd.read_csv(skill_df)
            logging.info("Read the dataframes")

            resume_df['Category'] = resume_df['Category'].apply(lambda x: self.lower_text(x))
            resume_df['Category_encoded'] = label_encoder.fit_transform(resume_df['Category'])
            resume_df['Resume'] = resume_df['Resume'].apply(lambda x: self.lower_text(x))
            resume_df['Resume'] = resume_df['Resume'].apply(lambda x: self.preprocessing(x))
            logging.info("Preprocessed the resume_df")

            skill_df['name'] = skill_df['name'].astype('str')
            skill_df['name'] = skill_df['name'].apply(lambda x: self.lower_text(x))
            related_lst = []
            for cols in skill_df.columns:
                if cols != 'name':
                    related_lst.append(cols)
            skill_df['related_list'] = list(skill_df.loc[:,related_lst].values)
            for cols in skill_df.columns:
                if cols != 'name' and cols != 'related_list':
                    skill_df.drop(cols,axis=1,inplace=True)
            logging.info("Preprocessed the skill_df")
            save_obj(file_path=self.data_transformation_config.preprocessed_resume_df_path,obj=resume_df)
            save_obj(file_path=self.data_transformation_config.preprocessed_skill_df_path,obj=skill_df)

            return (
                self.data_transformation_config.preprocessed_resume_df_path,
                self.data_transformation_config.preprocessed_skill_df_path
            )
        except Exception as e:
            raise CustomException(e,sys)