import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from http.client import HTTPException
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Query, Body
from sklearn.metrics import mean_squared_error
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_ingestion import fetch_weather_data, save_weather_data_to_db
from model.predict_series import predict, training_pipeline
from data.db_init import engine, get_engine
from data.db_class import Model, RealTemperature, Prediction


# Création des tables dans la base de données
models = [Model, RealTemperature, Prediction]
for model in models:
    model.metadata.create_all(bind=engine)

app = FastAPI(
    title="API Time series",
    description="API pour la prediction de séries temporelles",
    version="1.0.0",
)

# Configuration CORS pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # A remplacer par l'URL de votre frontend en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de prévision de séries temporelles"}


class DateRange(BaseModel):
    start_date: str
    end_date: str


@app.post("/fetch_data")
async def api_fetch_data(date_range: DateRange = Body(...)):
    start_date = date_range.start_date
    end_date = date_range.end_date
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Format de date invalide. Utiliser YYYY-MM-DD"
        )

    df = fetch_weather_data(start_date, end_date)
    msg = save_weather_data_to_db(df)

    return {"message": msg}


class TrainingParams(BaseModel):
    version: str
    start_date: str = None
    end_date: str = None


@app.post("/train_model")
async def train_model(params: TrainingParams = Body(...)):
    version = params.version
    start_date = params.start_date
    end_date = params.end_date

    if start_date and end_date:
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            return {"error": "Format de date invalide"}

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(RealTemperature)
        if start_date and end_date:
            query = query.filter(
                RealTemperature.timestamp >= start_date.strftime("%Y-%m-%d"),
                RealTemperature.timestamp <= end_date.strftime("%Y-%m-%d"),
            )

        results = query.all()

        # Convertir les résultats en DataFrame
        df = pd.DataFrame(
            [
                {
                    "timestamp": item.timestamp,
                    "temperature_2m": float(item.temperature_2m),
                    "relative_humidity": float(item.relative_humidity),
                    "precipitation": float(item.precipitation),
                    "surface_pressure": float(item.surface_pressure),
                    "latitude": float(item.latitude),
                    "longitude": float(item.longitude),
                }
                for item in results
            ]
        )

    finally:
        session.close()

    if df.empty:
        return {"error": "Aucune donnée disponible pour ces dates"}

    msg = training_pipeline(df, version)

    if msg:
        return {"message": "Modèle entraîné avec succès", "version": version}
    else:
        return {"error": "Erreur lors de l'entraînement du modèle"}


class PredictionRequest(BaseModel):
    model_id: int
    start_date: str
    end_date: str


@app.post("/predict")
async def prediction(request: PredictionRequest = Body(...)):
    model_id = request.model_id
    start_date = request.start_date
    end_date = request.end_date

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except ValueError:
        return {"error": "Format de date invalide"}

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    model = session.query(Model).filter(Model.id == model_id).first()
    if not model:
        session.close()
        return {"error": "Modèle non trouvé"}

    query = session.query(RealTemperature).filter(
        RealTemperature.timestamp >= start_date.strftime("%Y-%m-%d"),
        RealTemperature.timestamp <= end_date.strftime("%Y-%m-%d"),
    )
    results = query.all()
    session.close()
    if results:
        return {
            "error": "Les données sont déjà présentes dans la base de données. Veuillez choisir une autre période pour eviter un overfitting."
        }

    data = fetch_weather_data(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    results = predict(model.path, data)

    return results.to_dict(orient="records")


@app.get("/models", response_model=list)
async def get_models():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        models = session.query(Model).all()

        # Conversion des modèles en dictionnaires pour la réponse JSON
        models_list = [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "created_at": model.created_at,
                "path": model.path,
            }
            for model in models
        ]

        return models_list

    except Exception as e:
        return {"error": f"Erreur lors de la récupération des modèles: {str(e)}"}

    finally:
        session.close()


class PredictionDateRange(BaseModel):
    start_date: str
    end_date: str
    model_id: int = None  # Optionnel, permet de filtrer par modèle spécifique


@app.post("/predictions")
async def get_predictions(date_range: PredictionDateRange = Body(...)):
    try:
        start_date = pd.to_datetime(date_range.start_date)
        end_date = pd.to_datetime(date_range.end_date)
    except ValueError:
        return {"error": "Format de date invalide. Utilisez YYYY-MM-DD."}

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(Prediction).filter(
            Prediction.timestamp >= start_date.strftime("%Y-%m-%d"),
            Prediction.timestamp <= end_date.strftime("%Y-%m-%d"),
        )

        if date_range.model_id:
            query = query.filter(Prediction.model_id == date_range.model_id)

        predictions = query.all()

        if not predictions:
            return {
                "message": "Aucune prédiction n'est disponible pour cette période",
            }

        real_values = []
        pred_values = []
        predictions_list = []

        for pred in predictions:
            real = float(pred.real) if pred.real else None
            prediction = float(pred.prediction) if pred.prediction else None

            if real is not None and prediction is not None:
                real_values.append(real)
                pred_values.append(prediction)

            predictions_list.append(
                {
                    "id": pred.id,
                    "model_id": pred.model_id,
                    "timestamp": pred.timestamp,
                    "valeur_reelle": real,
                    "valeur_prevue": prediction,
                    "relative_humidity": (
                        float(pred.relative_humidity)
                        if pred.relative_humidity
                        else None
                    ),
                    "precipitation": (
                        float(pred.precipitation) if pred.precipitation else None
                    ),
                    "surface_pressure": (
                        float(pred.surface_pressure) if pred.surface_pressure else None
                    ),
                    "latitude": float(pred.latitude) if pred.latitude else None,
                    "longitude": float(pred.longitude) if pred.longitude else None,
                }
            )

        rmse = None
        if real_values and pred_values:
            rmse = float(np.sqrt(mean_squared_error(real_values, pred_values)))

        result = {}

        if rmse is not None:
            result["rmse"] = rmse

        result["predictions_count"] = len(predictions_list)
        result["predictions"] = predictions_list

        if date_range.model_id and predictions:
            model = session.query(Model).filter(Model.id == date_range.model_id).first()
            if model:
                result["model_info"] = {
                    "id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "created_at": model.created_at,
                }

        return result

    except Exception as e:
        return {"error": f"Erreur lors de la récupération des prédictions: {str(e)}"}

    finally:
        session.close()


import os
import subprocess


@app.get("/version")
async def get_version():
    """
    Retourne la version de l'API.

    En environnement local : "0.0.0"
    En environnement Docker : l'ID du dernier commit Git

    Returns:
        dict: Informations de version
    """
    # Vérifier si nous sommes dans un environnement Docker
    in_docker = os.path.exists("/.dockerenv")

    if in_docker:
        try:
            # En environnement Docker, essayer de récupérer l'ID du commit
            commit_id = os.environ.get("GIT_COMMIT_HASH", "unknown")

            # Si la variable d'environnement n'est pas définie, essayer de lire depuis un fichier
            if commit_id == "unknown" and os.path.exists("/app/git_commit_id.txt"):
                with open("/app/git_commit_id.txt", "r") as f:
                    commit_id = f.read().strip()

            return {"version": commit_id, "environment": "docker"}
        except Exception as e:
            return {"version": "unknown", "environment": "docker", "error": str(e)}
    else:
        # En environnement local
        return {"version": "0.0.0", "environment": "local"}
