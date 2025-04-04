import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Ajouter le répertoire parent au chemin Python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.predict_series import preprocess_data, training_pipeline, predict, create_features
from data.data_ingestion import fetch_weather_data, save_weather_data_to_db


class TestDataIngestion(unittest.TestCase):
    
    @patch('data.data_ingestion.requests.get')
    def test_fetch_weather_data_success(self, mock_get):
        # Configurer le mock pour simuler une réponse réussie
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                "temperature_2m": [10.5, 11.0],
                "relative_humidity_2m": [85, 87],
                "precipitation": [0, 0.2],
                "surface_pressure": [1015, 1014]
            },
            "hourly_units": {
                "temperature_2m": "°C",
                "relative_humidity_2m": "%",
                "precipitation": "mm",
                "surface_pressure": "hPa"
            }
        }
        mock_get.return_value = mock_response
        
        # Appeler la fonction avec des paramètres de test
        result = fetch_weather_data("2023-01-01", "2023-01-02")
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('temperature_2m', result.columns)
        self.assertIn('relative_humidity', result.columns)
        self.assertIn('precipitation', result.columns)
        self.assertIn('surface_pressure', result.columns)
        
    @patch('data.data_ingestion.requests.get')
    def test_fetch_weather_data_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_get.return_value = mock_response
        
        with self.assertRaises(Exception):
            fetch_weather_data("invalid_date", "2023-01-01")
            
    @patch('data.data_ingestion.get_engine')
    def test_save_weather_data_to_db(self, mock_get_engine):
        df = pd.DataFrame({
            'timestamp': [datetime.now(), datetime.now() + timedelta(hours=1)],
            'temperature_2m': [20.5, 21.0],
            'relative_humidity': [75, 78],
            'precipitation': [0, 0.1],
            'surface_pressure': [1010, 1012],
            'latitude': [48.8566, 48.8566],
            'longitude': [2.3522, 2.3522]
        })
        
        # Configurer le mock pour simuler une session SQLAlchemy
        mock_session = MagicMock()
        mock_session_maker = MagicMock(return_value=mock_session)
        mock_get_engine.return_value = MagicMock()
        
        with patch('data.data_ingestion.sessionmaker', return_value=mock_session_maker):
            result = save_weather_data_to_db(df)
            
        # Vérifier que la fonction renvoie un message de succès
        self.assertIn("enregistrements ajoutés", result)
        
        # Vérifier que add et commit ont été appelés sur la session
        self.assertTrue(mock_session.add.called)
        self.assertTrue(mock_session.commit.called)


class TestPredictSeries(unittest.TestCase):
    
    def setUp(self):
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=48, freq='h'),
            'temperature_2m': np.random.normal(20, 5, 48),
            'relative_humidity': np.random.normal(75, 10, 48),
            'precipitation': np.random.exponential(0.5, 48),
            'surface_pressure': np.random.normal(1010, 5, 48),
            'latitude': [48.8566] * 48,
            'longitude': [2.3522] * 48
        })
        
    
    # def test_preprocess_data(self):
    #     print(f"Test preprocess_data avec {(self.test_df)} lignes")
    #     X, y = preprocess_data(self.test_df)
        
    #     # Vérifier que X et y ne sont pas vides
    #     self.assertGreater(len(X), 0, "X ne devrait pas être vide")
    #     self.assertGreater(len(y), 0, "y ne devrait pas être vide")
    #     print(f"Après prétraitement: X a {len(X)} lignes et {len(X.columns)} colonnes")
    #     print(f"Colonnes de X: {X.columns.tolist()}")
        
    #     # Assertions
    #     self.assertIsInstance(X, pd.DataFrame)
    #     self.assertIsInstance(y, pd.Series)
    #     self.assertGreater(len(X.columns), 0)
    #     self.assertEqual(len(X), len(y))
    
    @patch('model.predict_series.get_engine')
    @patch('model.predict_series.joblib.dump')
    def test_training_pipeline(self, mock_dump, mock_get_engine):
        """Tester le pipeline d'entraînement"""
        # Configurer les mocks
        mock_session = MagicMock()
        mock_session_maker = MagicMock(return_value=mock_session)
        mock_get_engine.return_value = MagicMock()
        
        # Mocker la fonction preprocess_data pour qu'elle renvoie des données valides
        with patch('model.predict_series.preprocess_data') as mock_preprocess:
            # Créer des données de test pour X et y
            X_test = pd.DataFrame({
                'hour': [0, 3, 6, 9],
                'day_of_week': [0, 1, 2, 3],
                'month': [1, 1, 1, 1],
                'precipitation': [0.1, 0.0, 0.2, 0.0],
                'relative_humidity': [80, 75, 70, 65],
                'surface_pressure': [1010, 1012, 1015, 1013]
            })
            y_test = pd.Series([18.5, 20.0, 22.5, 21.0])
            mock_preprocess.return_value = (X_test, y_test)
            
            # Créer un répertoire temporaire pour les tests
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Patcher le chemin du modèle pour utiliser le répertoire temporaire
                with patch('model.predict_series.os.path.dirname', return_value=tmp_dir):
                    with patch('model.predict_series.sessionmaker', return_value=mock_session_maker):
                        # Exécuter le pipeline d'entraînement
                        result = training_pipeline(self.test_df, "test_version")
                        
                        # Vérifier que le modèle est enregistré
                        self.assertTrue(mock_dump.called)
                        self.assertTrue(mock_session.add.called)
                        self.assertTrue(mock_session.commit.called)
                        self.assertTrue(result)
    
    @patch('model.predict_series.joblib.load')
    def test_predict(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([20.5, 21.0])
        mock_load.return_value = mock_model
        
        # Créer un chemin de modèle fictif
        model_path = "model/registry/fake_model.pkl"
        
        # Mocker la fonction preprocess_data pour qu'elle renvoie des données valides
        with patch('model.predict_series.preprocess_data') as mock_preprocess:
            # Créer des données de test pour X et y
            X_test = pd.DataFrame({
                'hour': [0, 3],
                'day_of_week': [0, 1],
                'month': [1, 1],
                'precipitation': [0.1, 0.0],
                'relative_humidity': [80, 75],
                'surface_pressure': [1010, 1012]
            })
            y_test = pd.Series([18.5, 20.0])
            mock_preprocess.return_value = (X_test, y_test)
            
            # Mocks pour la base de données
            with patch('model.predict_series.get_engine'):
                with patch('model.predict_series.sessionmaker'):
                    with patch('model.predict_series.os.path.exists', return_value=True):
                        # Exécuter la fonction de prédiction avec un petit DataFrame
                        small_df = self.test_df.iloc[:5].copy()  # Prendre juste 5 lignes
                        results = predict(model_path, small_df)
                        
                        # Vérifier les résultats
                        self.assertIsInstance(results, pd.DataFrame)
                        self.assertIn('prediction', results.columns)
                        self.assertEqual(len(results), 2)  # 2 prédictions       


if __name__ == '__main__':
    unittest.main()