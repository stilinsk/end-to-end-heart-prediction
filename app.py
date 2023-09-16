
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import pickle
from models import Heart  # Import the Heart class from the models module

app = FastAPI()
model = pickle.load(open("trained_model.sav", "rb"))

@app.get("/{name}")
def hello(name):
    return {"message": f"Hello {name}! Welcome to this API."}

@app.get("/")
def greet():
    return {"message": "Hello, world!"}

@app.post("/predict")
def predict(req: Heart):
    features = [req.age, req.sex, req.cp, req.trestbps, req.chol, req.fbs, req.restecg,
                req.thalach, req.exang, req.oldpeak, req.slope, req.ca, req.thal]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]

    if prediction == 1:
        return {"ans": f"You have tested positive with a probability of {probability}."}
    else:
        return {"ans": f"You have tested negative with a probability of {probability}."}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)