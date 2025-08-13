from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.concurrency import asynccontextmanager
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# loader functions that you programmed!
from model import load_model, load_encoder
from pydantic import BaseModel
import pandas as pd
from typing import Annotated


class Person(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: float
    housing: str
    duration: int
    campaign: int


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["ohe"] = load_encoder()
    ml_models["models"] = load_model()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)
bearer = HTTPBearer()


@app.get("/")
async def root():
    return {"message": "Hello World"}


def get_username_for_token(token):
    if token == "abc123":
        return "pedro1"
    return None


async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials

    username = get_username_for_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"username": username}


@app.post("/predict")
async def predict(
    person: Annotated[
        Person,
        Body(
            examples=[
                {
                    "age": 29,
                    "job": "management",
                    "marital": "single",
                    "education": "unknown",
                    "balance": 560,
                    "housing": "no",
                    "duration": 459,
                    "campaign": 1,
                }
            ],
        ),
    ],
    user=Depends(validate_token),
):
    """
    Route to make predictions!
    """
    # Load the models
    ohe = ml_models["ohe"]
    model = ml_models["models"]

    df_person = pd.DataFrame([person.dict()])
    person_t = ohe.transform(df_person)
    pred = model.predict(person_t)[0]

    return {"prediction": str(pred), "username": user["username"]}
