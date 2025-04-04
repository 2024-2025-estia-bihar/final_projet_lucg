import os
import sys
from pathlib import Path
from datetime import datetime
from http.client import HTTPException
import pandas as pd

# Ajouter le répertoire parent au chemin Python (une seule fois suffit)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Maintenant que le chemin est correct, importer les modules de votre projet
from data.data_ingestion import fetch_weather_data, save_weather_data_to_db
from model.predict_series import predict, training_pipeline
from data.db_init import engine, get_engine
from data.db_class import Model, RealTemperature, Prediction
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware



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

# Inclusion des routeurs
from pydantic import BaseModel

class DateRange(BaseModel):
    start_date: str
    end_date: str

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de prévision de séries temporelles"}

@app.post("/fetch_data")
async def api_fetch_data(date_range: DateRange = Body(...)):
    start_date = date_range.start_date
    end_date = date_range.end_date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Format de date invalide. Utiliser YYYY-MM-DD")
        
    # Fonction pour récupérer les données météorologiques et les stocker dans la base de données
    df = fetch_weather_data(start_date, end_date)
    msg = save_weather_data_to_db(df)
    
    return {"message": msg}


class TrainingParams(BaseModel):
    version: str
    start_date: str = None
    end_date: str = None

@app.post("/train_model")
async def train_model(params: TrainingParams = Body(...)):
    # Extraire les paramètres du body
    version = params.version
    start_date = params.start_date
    end_date = params.end_date

    # Vérifier le format des dates
    if start_date and end_date:
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except ValueError:
            return {"error": "Format de date invalide"}

    # Créer une session SQLAlchemy
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Construire la requête en fonction des dates fournies
        query = session.query(RealTemperature)
        if start_date and end_date:
            query = query.filter(
                RealTemperature.timestamp >= start_date.strftime('%Y-%m-%d'),
                RealTemperature.timestamp <= end_date.strftime('%Y-%m-%d')
            )

        results = query.all()

        # Convertir les résultats en DataFrame
        df = pd.DataFrame([
            {
                'timestamp': item.timestamp,
                'temperature_2m': float(item.temperature_2m),
                'relative_humidity': float(item.relative_humidity),
                'precipitation': float(item.precipitation),
                'surface_pressure': float(item.surface_pressure),
                'latitude': float(item.latitude),
                'longitude': float(item.longitude)
            }
            for item in results
        ])
        
        # Vérifier la continuité de la serie temporelle
        
        
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
    # Extraire les paramètres du body
    model_id = request.model_id
    start_date = request.start_date
    end_date = request.end_date
    
    # Vérifier le format des dates
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except ValueError:
        return {"error": "Format de date invalide"}
    
    # Verifier que le modèle existe dans la base de données
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    
    model = session.query(Model).filter(Model.id == model_id).first()
    if not model:
        session.close()
        return {"error": "Modèle non trouvé"}
    
    # Recupérer les données météorologiques
    query = session.query(RealTemperature).filter(
        RealTemperature.timestamp >= start_date.strftime('%Y-%m-%d'),
        RealTemperature.timestamp <= end_date.strftime('%Y-%m-%d')
    )
    results = query.all()
    session.close()
    if results:
        return {"error": "Les données sont déjà présentes dans la base de données. Veuillez choisir une autre période pour eviter un overfitting."}
    
    data = fetch_weather_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    
    results = predict(model.path, data)
        
    return results.to_dict(orient='records')