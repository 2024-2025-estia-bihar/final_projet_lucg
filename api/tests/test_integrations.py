import unittest
from fastapi.testclient import TestClient
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au chemin Python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

# Importer l'application FastAPI
from api.main import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Bienvenue sur l'API de prévision de séries temporelles"},
        )

    @patch("api.main.fetch_weather_data")
    @patch("api.main.save_weather_data_to_db")
    def test_fetch_data_endpoint(self, mock_save, mock_fetch):
        # Configurer les mocks
        mock_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="h"),
                "temperature_2m": [20.0] * 24,
                "relative_humidity": [75.0] * 24,
                "precipitation": [0.0] * 24,
                "surface_pressure": [1010.0] * 24,
                "latitude": [48.8566] * 24,
                "longitude": [2.3522] * 24,
            }
        )
        mock_fetch.return_value = mock_df
        mock_save.return_value = "24 enregistrements ajoutés à la base de données"

        # Données pour la requête
        request_data = {"start_date": "2023-01-01", "end_date": "2023-01-02"}

        # Envoyer la requête
        response = self.client.post("/fetch_data", json=request_data)

        # Vérifier la réponse
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "24 enregistrements ajoutés à la base de données"},
        )

    @patch("api.main.get_engine")
    @patch("api.main.training_pipeline")
    def test_train_model_endpoint(self, mock_training, mock_get_engine):
        # Configurer les mocks pour simuler une base de données avec des données
        mock_session = MagicMock()
        mock_results = [
            MagicMock(
                timestamp="2023-01-01 00:00:00",
                temperature_2m="20.0",
                relative_humidity="75.0",
                precipitation="0.0",
                surface_pressure="1010.0",
                latitude="48.8566",
                longitude="2.3522",
            )
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = (
            mock_results
        )
        mock_session_maker = MagicMock(return_value=mock_session)
        mock_get_engine.return_value = MagicMock()

        with patch("api.main.sessionmaker", return_value=mock_session_maker):
            # Simuler un entraînement réussi
            mock_training.return_value = True

            # Données pour la requête
            request_data = {
                "version": "1.0.0",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
            }

            # Envoyer la requête
            response = self.client.post("/train_model", json=request_data)

            # Vérifier la réponse
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"message": "Modèle entraîné avec succès", "version": "1.0.0"},
            )

    @patch("api.main.get_engine")
    @patch("api.main.fetch_weather_data")
    @patch("api.main.predict")
    def test_predict_endpoint(self, mock_predict, mock_fetch, mock_get_engine):
        # Configurer les mocks
        mock_session = MagicMock()
        mock_model = MagicMock(id=1, path="model/registry/model_1.0.0.pkl")
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_model
        )
        mock_session.query.return_value.filter.return_value.all.return_value = (
            []
        )  # Pas de données existantes
        mock_session_maker = MagicMock(return_value=mock_session)
        mock_get_engine.return_value = MagicMock()

        # Simuler les données météo
        mock_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-02-01", periods=24, freq="h"),
                "temperature_2m": [20.0] * 24,
                "relative_humidity": [75.0] * 24,
                "precipitation": [0.0] * 24,
                "surface_pressure": [1010.0] * 24,
                "latitude": [48.8566] * 24,
                "longitude": [2.3522] * 24,
            }
        )
        mock_fetch.return_value = mock_df

        # Simuler les prédictions
        mock_predict_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2023-02-01", periods=24, freq="h"),
                "prediction": [21.5] * 24,
            }
        )
        mock_predict.return_value = mock_predict_df

        with patch("api.main.sessionmaker", return_value=mock_session_maker):
            # Données pour la requête
            request_data = {
                "model_id": 1,
                "start_date": "2023-02-01",
                "end_date": "2023-02-02",
            }

            # Envoyer la requête
            response = self.client.post("/predict", json=request_data)

            # Vérifier la réponse
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json(), list)
            self.assertEqual(len(response.json()), 24)


if __name__ == "__main__":
    unittest.main()
