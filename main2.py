import pickle


import joblib
random_forest = pickle.load(open("random_forest.pkl","rb"))
StandartScaler = pickle.load(open("standart_scaler.pkl","rb"))

veri = [2.000,117,90,19,71,25.2,0.313,21]

scaled_veri = StandartScaler.transform(veri)