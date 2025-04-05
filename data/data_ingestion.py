import requests
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from datetime import datetime
from typing import Optional, Union
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from data.db_class import RealTemperature
from data.db_init import get_engine


def fetch_weather_data(
    start_date: Union[str, datetime], end_date: Optional[Union[str, datetime]] = None
) -> pd.DataFrame:

    api_url = "https://archive-api.open-meteo.com/v1/archive"

    longitude = 2.3522
    latitude = 48.8566

    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    elif isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
        ],
        "timezone": "auto",
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(data["hourly"]["time"]),
                "temperature_2m": data["hourly"]["temperature_2m"],
                "relative_humidity": data["hourly"]["relative_humidity_2m"],
                "precipitation": data["hourly"]["precipitation"],
                "surface_pressure": data["hourly"]["surface_pressure"],
                "latitude": latitude,
                "longitude": longitude,
            }
        )

        if df.isnull().values.any():
            for col in df.columns:
                if col != "timestamp" and df[col].isnull().any():
                    df[col] = df[col].interpolate(method="linear")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération des données: {e}")
        return pd.DataFrame()


def save_weather_data_to_db(df: pd.DataFrame) -> bool:

    if df.empty:
        print("Aucune donnée à enregistrer")
        return False

    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()

        records_added = 0

        for _, row in df.iterrows():
            timestamp_str = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

            new_record = RealTemperature(
                timestamp=timestamp_str,
                temperature_2m=str(row["temperature_2m"]),
                relative_humidity=str(row["relative_humidity"]),
                precipitation=str(row["precipitation"]),
                surface_pressure=str(row["surface_pressure"]),
                latitude=str(row["latitude"]),
                longitude=str(row["longitude"]),
            )

            try:
                session.add(new_record)
                session.commit()
                records_added += 1
            except IntegrityError:
                session.rollback()
                continue

        session.close()
        return f"{records_added} enregistrements ajoutés à la base de données"

    except Exception as e:
        return f"Erreur lors de l'enregistrement des données: {e}"
