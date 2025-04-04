import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Utiliser des chemins absolus pour garantir la cohérence
BASE_DIR = Path(__file__).parent.parent  # Remonter au répertoire racine du projet
DATA_DIR = BASE_DIR / "data"

# S'assurer que le répertoire data existe
os.makedirs(DATA_DIR, exist_ok=True)

# Utiliser un chemin absolu pour la base de données
DB_PATH = os.path.join(DATA_DIR, "sql_app.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_engine():
    return engine
