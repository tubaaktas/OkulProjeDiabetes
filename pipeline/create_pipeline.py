def preprocess_data_(data):
    if data[0] < 35:
        data.append('young')
    if 35 <= data[0] <= 55:
        data.append('middleage')
    else:
        data.append('old')

    if data[1] < 18.5:
        data.append('underweight')
    elif 18.5 <= data[1] < 24.9:
        data.append('normal')
    elif 24.9 <= data[1] < 29.9:
        data.append('overweight')
    else:
        data.append('obese')

    if data[2] < 140:
        data.append('Normal')
    elif 140 <= data[2] < 200:
        data.append('Prediabetes')
    else:
        data.append('Diabetes')

    if data[3] < 79:
        data.append('normal')
    elif 79 <= data[3] < 89:
        data.append('hs1')
    else:
        data.append('hs2')

    if 16 <= data[4] <= 166:
        data.append('Normal')
    else:
        data.append('Abnormal')
        # Burayı böyle doldurmalısın, data[4] tamamen rastgele yazdığım bir şey


import joblib
import pandas as pd
import numpy as np

data = pd.read_csv('X_train.csv')

model = joblib.load('model.joblib')
model.predict_proba(np.array(data.iloc[86]).reshape(1, -1))
