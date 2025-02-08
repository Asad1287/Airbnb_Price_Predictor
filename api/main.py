from fastapi import FastAPI
from pydantic import BaseModel
from src.models.inference import load_model, predict
import os

app = FastAPI(
    title="Titanic ML API",
    description="API for predicting Titanic survival",
    version="1.0"
)

# Load the trained model at startup.
MODEL_PATH = os.path.join("models", "titanic_model.pkl")
model = load_model(MODEL_PATH)

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic ML API!"}

@app.post("/predict")
def make_prediction(passenger: Passenger):
    # Convert the input to the expected feature list.
    # For simplicity, assume Sex: male=1, female=0 and Embarked: C=0, Q=1, S=2.
    sex_numeric = 1 if passenger.Sex.lower() == "male" else 0
    embarked_mapping = {"C": 0, "Q": 1, "S": 2}
    embarked_numeric = embarked_mapping.get(passenger.Embarked.upper(), 2)
    features = [
        passenger.Pclass,
        sex_numeric,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        embarked_numeric
    ]
    prediction = predict(model, features)
    return {"prediction": int(prediction)}
