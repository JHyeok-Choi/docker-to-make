#!/usr/bin/env python
# coding=utf-8

from fastapi import FastAPI, status, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

### MYSQL RDS
from pydantic import BaseModel, EmailStr, Field
from database import db_conn
from models import St_info, St_grade

### Kakao geoLocation
import requests
import hashlib
import hmac
import base64
import time


from dotenv import load_dotenv
import os
load_dotenv()

ACCESSKEY = os.environ.get('AccessKey')
SECRETKEY = os.environ.get('SecretKey')
IP=os.environ.get('IP')


from typing_extensions import Annotated

import sys
import contextlib
import requests



import os
import librosa
import librosa.display
import struct 
import numpy as np
import IPython.display as ipd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint  
from keras.models import load_model




import soundfile
import io
import pydub
import librosa


######################
# ENCODING
from pprint import pprint
import scipy.io.wavfile
import numpy

######################



"""
$ export FLASK_APP=urban_sound_classifier.py
$ flask run
... Running on http://127.0.0.1:5000

"""
#
# model = load_model('urban_sound_model.h5')
# print("Model loaded from urban_sound_model.h5")
#
fulldatasetpath = '../urbansund8k/'

#
# def extract_feature(file_name):
#     try:
#         audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#         mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
#         mfccsscaled = np.mean(mfccs.T, axis=0)
#     except Exception as e:
#         print("Error encountered while parsing file:", file_name)
#         return None
#     return np.array([mfccsscaled])


def make_signature(method, basestring, timestamp, access_key, secret_key):

    message = method = " " + basestring + "\n" + timestamp + "\n" + access_key
    secret_key = bytes(secret_key, 'UTF-8')
    message = bytes(message, 'UTF-8')
    signature = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())


    return signature


def requestApi(timestamp, access_key, signature, uri):
    print('t: ', timestamp)
    headers = {'x-ncp-apigw-timestamp': timestamp,
                'x-ncp-iam-access-key': access_key,
                'x-ncp-apigw-signature-v2': signature}
    
    res = requests.get(uri, headers=headers)

    print('status : %d' % res.status_code)
    print('content : %s' % res.content)


metadata = pd.read_csv('../urbansound8k/UrbanSound8K.csv')
le = LabelEncoder()
le.fit(metadata['class'])


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = db_conn()
session = db.sessionmaker()


@app.get(
    path='/', description="main",
    responses={200:{"description": "okok"}}
)
async def open_main():
    return "OK"

if __name__ == "__main__":
    uvicorn.run (app, host="0.0.0.0", port=3500)

@app.post('/model_test')
async def model_test(file: UploadFile):
    #result = session.query(St_info)
    
    filename = file.filename
    test_file = os.path.join('./upload', filename)
    content = await file.read()
    with open(os.path.join('./upload', filename), 'wb') as fp:
            fp.write(content)
    print("check file: ", file)
    # test_feature = extract_feature(test_file)
    #dB = librosa.power_to_db(test_feature, ref=np.max)

    y, sr = librosa.load(test_file)
    S = np.abs(librosa.stft(y))

    dB = librosa.amplitude_to_db(S, ref=1e-05)
    #dB = librosa.amplitude_to_db(test_feature, ref=1e-05)

    print("check test_decibel: ", dB)

    # scaler01 = MinMaxScaler(feature_range=(0, 80))
    # scaler01.fit(dB)
    # scaler01_scaled = scaler01.transform(dB)
    #cdB = dB + 80

    #print('convert dB: ', cdB)

    #print("check dB: ", np.mean(cdB))
    print("check dB argmax: ", np.argmax(dB))
    print("check dB max: ", np.max(dB))
    print("check dB argmin: ", np.argmin(dB))
    print("check db min: ", np.min(dB))
    print("check db mean: ", np.mean(dB))

    print("datetimme: ", datetime.now())
    #print("TIMESTAMP: ", TIMESTAMP)

    # print("longitude: ", 123)
    # print("latitude: ", 456)

    
    # #secret_key = bytes(SECRETKEY, 'UTF-8')
    
    # method= "GET"
    # baseuri = f"/geolocation/v2/geoLocation?ip={IP}&ext=t&responseFormatType=json"
    # timestamp = str(int(time.time() * 1000))
    # signature = make_signature(method, baseuri, timestamp, ACCESSKEY, SECRETKEY)


    # hostname = "https://geolocation.apigw.ntruss.com"
    # requestUri = hostname + baseuri

    # print(timestamp)
    # print(signature)
    # print(requestUri)
    # print(ACCESSKEY)
    # print(SECRETKEY)

    # requestApi(timestamp, ACCESSKEY, signature, requestUri)



    # print("status: ", res.status_code)
    # print("data: ", res.json)

    # ## to graph
    # plt.plot(y)
    # plt.savefig('test011_1.png')

    # plt.plot(S)
    # plt.savefig('test021_1.png')

    # if file:
    #     filename = file.filename
    #     test_file = os.path.join('./upload', filename)
    #     #file.save(audio_file)
    #     content = await file.read()
    #     with open(os.path.join('./upload', filename), 'wb') as fp:
    #         fp.write(content)

    #     # mfccs = extract_feature(audio_file)
    #     # test_file = 'drill.wav'
    #     print("check test_file: ", test_file)
    #     print("check test_decibel: ", librosa.power_to_db(test_file, ref=np.max))
    #     test_feature = extract_feature(test_file)
    #     print("check feature: ", test_feature)

    #     global why
    #     why = ''
    #     if test_feature is not None:
    #         predicted_proba_vactor = model.predict(test_feature)
    #         predicted_class_index = np.argmax(predicted_proba_vactor)
    #         predicted_class_label = le.inverse_transform([predicted_class_index])[0]
    #         why = predicted_class_label
    #         print(f"The predicted class for {test_file} is: {predicted_class_label}")
    #     else:
    #         print("Failed to extract features from the file.")

    #     return f'{why}'

    return "OKOK Done"


@app.post('/blob2file')
async def B2F(file: UploadFile):

    a = bytearray(file.file.read())
    print('1 : ', a)
    b = numpy.array(a, dtype=numpy.int16)
    print('2 : ', b)
    soundfile.write('test000.wav', b, 48000, format='WAV')


@app.get('/getuser')
async def getuser(id=None, name=None):
    if (id is None) and (name is None):
        return "학번 또는 이름으로 검색하세요."
    else:
        if name is None:
            result = session.query(St_info).filter(St_info.ST_ID == id).all()
        elif name is None:
            result = session.query(St_info).filter(St_info.NAME == name).all()
        else:
            result = session.query(St_info).filter(St_info.ST_ID == id, St_info.NAME == name).all()
        return result

@app.get('/stinfo')
async def select_st_info():
    result = session.query(St_info)
    return result.all()