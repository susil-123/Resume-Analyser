import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from src.utils import load_object
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from src.logger import logging
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass
        
    def preprocessing(self,text):
        text = text.lower()
        words = word_tokenize(text,language='english')
        words = [lemmatizer.lemmatize(word,pos='v') for word in words if word not in stopwords.words('english')]
        words = [i.replace(',','') for i in words]
        words = [i.replace('.','') for i in words]
        words = [i.replace('*','') for i in words]
        words = [i.replace('-','') for i in words]
        words = [i.replace(':','') for i in words]
        return ' '.join(words)

    def predict(self,features):
        try:
            dict = {6: 'data science',
            12: 'hr',
            0: 'advocate',
            1: 'arts',
            24: 'web designing',
            16: 'mechanical engineer',
            22: 'sales',
            14: 'health and fitness',
            5: 'civil engineer',
            15: 'java developer',
            4: 'business analyst',
            21: 'sap developer',
            2: 'automation testing',
            11: 'electrical engineering',
            18: 'operations manager',
            20: 'python developer',
            8: 'devops engineer',
            17: 'network security engineer',
            19: 'pmo',
            7: 'database',
            13: 'hadoop',
            10: 'etl developer',
            9: 'dotnet developer',
            3: 'blockchain',
            23: 'testing'}
            model_path=os.path.join("artifacts","model.pkl")
            vector_path=os.path.join('artifacts','vectorizer.pkl')
            
            model=load_object(file_path=model_path)
            vectorizer=load_object(file_path=vector_path)
            logging.info("Model and vectorizer loaded")
            
            pre_txt = self.preprocessing(features)
            text = vectorizer.transform([pre_txt])
            sample_pred = pd.DataFrame(text.toarray(),columns=vectorizer.get_feature_names_out())
            pred = model.predict(sample_pred)
            logging.info("Model predicted the custom data")

            return dict[pred[0]]
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_related(self,sample):
        try:
            resume_dict = {}
            skill_df_path=os.path.join("artifacts","preprocessed_skill_df.pkl")
            skill_df = load_object(file_path=skill_df_path)
            logging.info("skill df loaded")

            val = self.predict(sample)
            row = skill_df[skill_df['name'] == val]
            if len(row) < 1:
                # print('no results found')
                resume_dict['domain'] = 'no result'
            else:
                # print(f"Domain: {row['name'].values[0]}\n")
                resume_dict['domain'] = row['name'].values[0]
                not_matched = []
                matched = []
                related_list = row['related_list'].values[0]
                related_list = np.delete(related_list,0)
                for i in related_list:
                    pattern = re.compile(re.escape(i), re.IGNORECASE)
                    match = re.search(pattern, sample)
                    if not match:
                        not_matched.append(i)
                    else:
                        matched.append(i)
                # print(not_matched)
                resume_dict['recommended'] = not_matched
                resume_dict['found'] = matched
                return resume_dict
                
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_output(self,pdf_path):
        try:
            # pdf_path = os.path.join('data','resume2.pdf')

            poppler_path = r'C:\\Users\\susil\\Downloads\\Release-24.02.0-0\\poppler-24.02.0\\Library\\bin'
            pages = convert_from_path(pdf_path, 300, poppler_path=poppler_path)
            os.makedirs('pdf_images', exist_ok=True)

            image_files = []
            for i, page in enumerate(pages):
                image_path = f'pdf_images/page_{i + 1}.jpg'
                page.save(image_path, 'JPEG')
                image_files.append(image_path)
            logging.info("pdf images saved on to pdf_images folder")

            pdf_text = ""
            for image_file in image_files:
                text = pytesseract.image_to_string(Image.open(image_file))
                pdf_text += text + "\n"
            logging.info("pdf images converted into texts")

            # Print or save the extracted text
            return self.get_related(pdf_text)
        except Exception as e:
            raise CustomException(e,sys)