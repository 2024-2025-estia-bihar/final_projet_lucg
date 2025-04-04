from data.db_init import get_engine
from sqlalchemy.orm import sessionmaker
from data.db_class import Prediction, Model
from sqlalchemy.exc import IntegrityError
import joblib
import os
from datetime import datetime
from data.db_init import get_engine
from sqlalchemy.orm import sessionmaker
from data.db_class import Model
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def create_features(df):
    df_features = df.copy()
    
    # Création des variables retardées (lag variables)
    for i in range(1, 20):
        df_features[f'temp_lag_{i}'] = df_features['temperature_2m'].shift(i)
    
    # Variables temporelles
    df_features['hour'] = df_features.index.hour
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['day'] = df_features.index.day
    
    # moyenne mobile
    df_features['temp_rolling_mean_24h'] = df_features['temperature_2m'].rolling(window=8).mean()  # Moyenne mobile sur 24h (8 périodes de 3h)
    df_features['temp_rolling_std_24h'] = df_features['temperature_2m'].rolling(window=8).std()  # Écart-type mobile sur 24h
    
    df_features = df_features.dropna()
    
    return df_features

def preprocess_data(df):
    data = df.copy()
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    data['hour'] = data['timestamp'].dt.hour
    data['hour_group'] = (data['hour'] // 3) * 3
    data['date'] = data['timestamp'].dt.date

    data_3h = data.groupby(['date', 'hour_group']).agg({
        'temperature_2m': 'mean',
        'relative_humidity': 'mean',
        'precipitation': 'mean',
        'surface_pressure': 'mean'
    }).reset_index()

    data_3h['timestamp'] = pd.to_datetime(data_3h['date'].astype(str) + ' ' + data_3h['hour_group'].astype(str) + ':00:00')

    data_3h = data_3h[['timestamp', 'temperature_2m', 'relative_humidity', 'precipitation', 'surface_pressure']]
    data_3h.set_index('timestamp', inplace=True)

    
    # Ajoute de variables artificielles/exogenes
    data_3h = create_features(data_3h)
    
    X = data_3h.drop(columns=['temperature_2m'])
    y = data_3h['temperature_2m']
    
    return X, y

def train_model(X, y, version):
    
    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20, min_samples_split=2)
    model.fit(X, y)
    
    # Préparer les métadonnées
    model_path = f'model/registry/model{version}.pkl'
    model_name = 'RandomForestRegressor'
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Sauvegarder le fichier du modèle
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        model_entry = Model(
            name=model_name,
            version=version,
            created_at=created_at,
            path=model_path
        )
        
        session.add(model_entry)
        session.commit()
        
        model_id = model_entry.id
        session.close()
        
        print(f"Modèle enregistré avec ID {model_id}")
        
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du modèle dans la base de données: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
    
    return model_path

def predict(path, X_input):
    
    # Vérifier que le modèle existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier de modèle n'existe pas: {path}")
    
    # Charger le modèle enregistré
    model = joblib.load(path)
    
    # Copie des données d'entrée
    X_processed = X_input.copy()
    
    # Prétraiter les données comme à l'entraînement
    X, y = preprocess_data(X_processed)
    
    # Faire les prédictions
    y_pred = model.predict(X)
    
    # Créer un DataFrame de résultats avec des colonnes de même longueur
    results = []
    
    # Récupérer la latitude et la longitude (qui sont constantes)
    latitude = X_input['latitude'].iloc[0] if 'latitude' in X_input.columns else None
    longitude = X_input['longitude'].iloc[0] if 'longitude' in X_input.columns else None
    
    # Créer une ligne de résultat pour chaque prédiction
    for i in range(len(y_pred)):
        # Créer un dictionnaire pour chaque ligne
        row = {
            'prediction': y_pred[i],
            'timestamp': X.index[i] if hasattr(X, 'index') and len(X.index) == len(y_pred) else None,
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Ajouter les colonnes si elles existent dans X
        for col in ['relative_humidity', 'precipitation', 'surface_pressure']:
            if col in X.columns:
                row[col] = X[col].iloc[i]
        
        # Ajouter la latitude et la longitude (constantes)
        row['latitude'] = latitude
        row['longitude'] = longitude
        
        # Ajouter la valeur réelle si elle existe
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            if len(y) == len(y_pred):
                row['real'] = y.iloc[i]
        
        results.append(row)
    
    # Créer un DataFrame à partir des résultats
    result_df = pd.DataFrame(results)
    
    # Enregistrer les prédictions dans la base de données
    
    try:
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Récupérer l'ID du modèle
        model_info = session.query(Model).filter(Model.path == path).first()
        model_id = model_info.id if model_info else None
        
        if model_id:
            # Enregistrer chaque prédiction dans la base de données
            for _, row in result_df.iterrows():
                prediction_entry = Prediction(
                    model_id=model_id,
                    timestamp=str(row['timestamp']) if 'timestamp' in row else None,
                    relative_humidity=str(row['relative_humidity']) if 'relative_humidity' in row else None,
                    precipitation=str(row['precipitation']) if 'precipitation' in row else None,
                    surface_pressure=str(row['surface_pressure']) if 'surface_pressure' in row else None,
                    latitude=str(row['latitude']) if 'latitude' in row else None,
                    longitude=str(row['longitude']) if 'longitude' in row else None,
                    prediction=str(row['prediction']),
                    real=str(row['real']) if 'real' in row else None
                )
                
                try:
                    session.add(prediction_entry)
                    session.commit()
                except IntegrityError:
                    # En cas de doublon, faire un rollback et continuer
                    session.rollback()
        
        session.close()
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des prédictions: {e}")
    
    return result_df


def training_pipeline(data, version):
    
    df = data.copy()

    X, y = preprocess_data(df)
    
    model = train_model(X, y, version)
    
    return f"Model trained and saved at {model}"