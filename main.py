from fastapi import FastAPI
import joblib
import numpy as np
import os
from pydantic import BaseModel

# Inicializa a API
app = FastAPI()

class InputData(BaseModel):
    features : list

# Carrega o modelo salvo
modelo = joblib.load("modelo.pkl")

@app.get("/")
def home():
    return {"message": "API Decision Tree rodando!"}

@app.post("/predict")
def predict(data : InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = modelo.predict(arr)
    return {"prediction": int(pred[0])}

