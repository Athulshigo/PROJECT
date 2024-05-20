from django.shortcuts import render
import os
import time
import random

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

# import matplotlib.pyplot as plt
import re
import gc

from rest_framework.decorators import api_view
from rest_framework.response import Response
import io
from django.core.files.base import ContentFile
from io import BytesIO
from .models import File
import uuid
import requests
# Create your views here.

def home(request):
    return render(request, 'index.html')

def generate_test_predictions(model, test):
    """ Takes fit XGBOOST model and test data as inputs. 
        Generates and returns model predictions on test data.
    """
    predictions = np.expm1(model.predict(test.values))
    return predictions  

def generate_test_predictions(model, test):
    """ Takes fit XGBOOST model and test data as inputs. 
        Generates and returns model predictions on test data.
    """
    predictions = np.expm1(model.predict(test.values))
    return predictions  

def generate_submission_CSV_predictions(predictions, model_name): 
    """ Takes predictions, validation RMSLE score, string of model name, and index used for validation set as inputs. 
        Writes test predictions to CSV file.
    """             
    # Create dataframe with test predictions
    df = pd.DataFrame(predictions, dtype = float, columns = ['Demanda_uni_equil'])      
    
    # Write to csv
    # os.chdir('D:/OneDrive/Documents/Kaggle/Grupo Bimbo Inventory Demand/submissions')  
    df.sort_values(by='Demanda_uni_equil', ascending=False, inplace=True)
    name = 'submission_' + model_name + '_'+ '.csv'
    # df.to_csv(name, index = True, header = True, index_label = 'id')
    # os.chdir('D:/OneDrive/Documents/Kaggle/Grupo Bimbo Inventory Demand/data')  

    stream = io.BytesIO()
    df.to_csv(stream, index=True, header=True, index_label='id')
    stream.seek(0)
    
    return stream,name

def savefile(name, output_file):
    """ Takes string of file name and output file as inputs. 
        Saves output file to disk.
    """
    unique_id = str(uuid.uuid4())
    key_name = f"{name}_{unique_id}"
    output_file.seek(0)
    report_model = File()

    report_model.title = key_name
    report_model.file.save(key_name+".csv", ContentFile(output_file.read()))
    report_model.save()

    print("Report files saved successfully.")

    output_file.close()

    url = report_model.file.url
     
    return url
@api_view(['POST'])
def predict(request):
    """ Takes POST request with test data as input. 
        Generates and returns model predictions on test data.
    """
    # Load model
    url = "https://salespredict.ams3.cdn.digitaloceanspaces.com/salesPredict/xgboost_model.pkl"
    response = requests.get(url)
    try:
        model = pickle.loads(response.content)
        print("model loaded successfully")
    except Exception as e:
        return Response({'error': str(e)})
    test_types = {
         'Semana': np.uint8, 'Agencia_ID': np.uint16
         , 'Canal_ID': np.uint8
         , 'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16
         , 'Client_Type': np.uint32
         , 'short_product_name': np.uint32, 'weight': np.float32, 'pieces': np.float32, 'weight_per_piece': np.float32 
         , 'Demanda_uni_equil_tminus2': np.float16, 'Demanda_uni_equil_tminus3': np.float16
         , 'Demanda_uni_equil_tminus4': np.float16, 'Demanda_uni_equil_tminus5': np.float16          
         , 'Agencia_ID_count': np.float32, 'Canal_ID_count': np.float32, 'Ruta_SAK_count': np.float32
         , 'Cliente_ID_count': np.float32, 'Producto_ID_count': np.float32, 'Client_Type_count': np.float32
         }  
    # Load test data
    try:
        test = pd.read_csv(request.data['test'],sep = ",", usecols = test_types.keys(), dtype = test_types)
        print(test.head())
        test = test.reset_index(drop=True)    
        print("data loaded successfully")
    except Exception as e:
        return Response({'error': str(e)})
    
    # Generate predictions
    try:
        predictions = generate_test_predictions(model, test)
        print("predictions generated successfully")
    except Exception as e:
        return Response({'error': str(e)})
    try:
        stream, name = generate_submission_CSV_predictions(predictions, model_name = 'xgb_pickle_loaded')
        url = savefile(name, stream) 
        print("predictions saved successfully")
        return Response({'url': url,'message':'predictions generated successfully'})
    except Exception as e:
        return Response({'error': str(e)})

