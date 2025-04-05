# API Time Series et Classification Multimodale

Ce projet combine deux composantes majeures : une API de prédiction de séries temporelles météorologiques et un système de classification multimodale de textes. L'objectif est de fournir une solution complète pour l'analyse et la prédiction de données temporelles, ainsi que la classification de données textuelles.

## Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Pipeline CI/CD](#pipeline-cicd)
- [Installation](#installation)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Contributeurs](#contributeurs)
- [Licence](#licence)

## Description

L'API de séries temporelles est basée sur FastAPI et utilise des modèles de machine learning pour prédire des données météorologiques futures.

## Fonctionnalités

- **Récupération de données météorologiques historiques**
- **Entraînement de modèles prédictifs** sur une periode donnée
- **Génération de prédictions futures** basées sur les modèles entraînés
- **Évaluation de la performance** des modèles via des métriques (RMSE)
- **Interface REST** pour l'intégration facile avec d'autres systèmes

## Architecture

Le projet est structuré selon une architecture modulaire :

### Couche d'accès aux données (data)
- `data_ingestion.py` : Contient les fonctions pour recupérer les données météorologiques depuis une API externe
- `db_init.py` : Initialise la base de données SQLite
- `db_class.py` : Définit les modèles ORM SQLAlchemy

### Couche modèle (model)
- `predict_series.py` : Contient le pipeline d'entraînement et de prédiction des modèles de séries temporelles

### Couche API (api)
- `main.py` : Points d'entrée API FastAPI
- `tests/` : Tests unitaires et d'intégration


## Pipeline CI/CD

Le pipeline CI/CD est configuré pour automatiser les tests et le déploiement de l'API. Il utilise GitHub Actions pour exécuter les tests à chaque push sur les branches master ou develop.

### 1er Etape : Lint

Le code est analysé pour détecter les erreurs de syntaxe et les problèmes de style. Des outils comme `flake8` et `black` sont utilisés pour garantir la qualité du code.

### 2e Etape : Tests

Des tests unitaires et d'intégration sont exécutés pour valider le bon fonctionnement de l'API. Les tests sont écrits avec `pytest` et couvrent les principales fonctionnalités de l'API. Ils sont disponibles dans le répertoire `api/tests/`.

### 3e Etape : Versionning

Cette étape génère une nouvelle version de l'application basée sur le dernier commit Git. Un identifiant unique est généré à partir du hash du commit et utilisé pour créer un tag d'image Docker. 

### 4e Etape : Build et deploiement

L'image Docker est construite et déployée sur un registre Docker (Github package). On s'assure que la derniere image ait le tag `latest` en plus du tag de version.

## Installation

### Prérequis
- Docker et Docker Compose installés
- Git pour cloner le dépôt

### Étapes d'installation
1. Cloner le dépôt
   ```bash
   git clone https://github.com/2024-2025-estia-bihar/final_projet_lucg.git
   cd final_projet_lucg
   ```

2. Connectez-vous à GitHub Packages :
    ```bash
    echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
    ```
    Remplacez `CR_PAT` par votre token d'accès personnel GitHub et `USERNAME` par votre nom d'utilisateur GitHub. L'utilisateur doit avoir les droits d'accès au registre de packages.

3. Construire et démarrer les conteneurs Docker
   ```bash
   docker-compose up -d
   ```

Une fois les conteneurs démarrés, l'API est accessible à l'adresse http://localhost:8000.

## Utilisation de l'API

### 1. Documentation interactive
Accédez à la documentation interactive Swagger UI :
```
http://localhost:8000/docs
```

### 2. Récupération de données météorologiques
```bash
curl -X POST "http://localhost:8000/fetch_data" -H "Content-Type: application/json" -d '{"start_date": "2022-01-01", "end_date": "2024-12-31"}'
```

### 3. Entraînement d'un modèle
```bash
curl -X POST "http://localhost:8000/train_model" -H "Content-Type: application/json" -d '{"version": "1.0.0", "start_date": "2025-01-01", "end_date": "2025-01-31"}'
```

### 4. Liste des modèles disponibles
```bash
curl -X GET "http://localhost:8000/models"
```

### 5. Génération de prédictions
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"model_id": 1, "start_date": "2025-01-01", "end_date": "2025-01-31"}'
```

### 6. Récupération des prédictions stockées avec RMSE
```bash
curl -X POST "http://localhost:8000/predictions" -H "Content-Type: application/json" -d '{"model_id": 1, "start_date": "2025-01-01", "end_date": "2025-01-31"}'
```

## Contributeurs

Projet réalisé par LucG Mensah dans le cadre du projet final 2024-2025 ESTIA Bihar.
Référentiel GitHub: [2024-2025-estia-bihar/final_projet_lucg](https://github.com/2024-2025-estia-bihar/final_projet_lucg)

## Licence

Ce projet est distribué sous licence MIT.
