FROM python:3.11-slim

WORKDIR /app

# Arguments pour capturer l'ID du commit
ARG GIT_COMMIT_HASH="unknown"

# Stocker l'ID du commit comme variable d'environnement et dans un fichier
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}
RUN echo ${GIT_COMMIT_HASH} > /app/git_commit_id.txt

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement le fichier requirements.txt d'abord
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Créer les répertoires nécessaires et s'assurer qu'ils existent
RUN mkdir -p model/registry data logs

# Ajouter le répertoire courant au PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]