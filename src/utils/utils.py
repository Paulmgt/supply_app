import json
import os
from dotenv import load_dotenv
import threading
from tqdm import tqdm
import asyncio
import numpy as np
from hashlib import sha256
import pandas as pd 
from typing import List
from sqlalchemy.orm import scoped_session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from src.users.models import UserIn, UserOut, CleanedData, SessionLocal
from src.clean.clean_a import f_data_clean
from src.clean.clean_b import f_data_clean_2


# Charger les variables d'environnement du fichier .env
load_dotenv()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Obtenir la clé secrète à partir de l'environnement
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")
ALGORITHM = "HS256"
JSON_FILE_PATH = os.path.expanduser("src/users/users.json")

def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def load_users() -> List[UserIn]:
    if not os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "w") as file:
            json.dump([], file)
    
    with open(JSON_FILE_PATH, "r") as file:
        users_data = json.load(file)
    
    return [UserIn(**user) for user in users_data]

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    try:
        # Décodez le token et obtenez le nom d'utilisateur
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(status_code=401, detail="Token invalide")
        
        # Chargez tous les utilisateurs
        users = load_users()
        
        # Trouvez l'utilisateur correspondant
        user = next((user for user in users if user.username == username), None)
        
        if user is None:
            raise HTTPException(status_code=401, detail="Utilisateur non trouvé")
        
        return UserOut(**user.dict())  # Assurez-vous que UserOut est correctement défini

    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

def save_user(user):
    users = load_users()
    users.append(user)
    with open(JSON_FILE_PATH, "w") as file:
        json.dump([user.dict() for user in users], file)


def get_last_entry(session):
    """Récupère la dernière entrée en fonction de la date du commentaire et du titre."""
    try:
        last_entry = session.query(CleanedData).order_by(CleanedData.date_commentaire.desc()).first()
        return last_entry.date_commentaire, last_entry.titre_com if last_entry else (None, None)
    except SQLAlchemyError as e:
        print(f"Erreur lors de la récupération de la dernière entrée: {e}")
        return None, None


def insert_data_thread(df_chunk, session_factory, total_rows, pbar, thread_id):
    """Insère les données d'un chunk dans la base de données via un thread, met à jour la barre de progression."""
    try:
        session = session_factory()
        print(f"[Thread {thread_id}] Démarrage avec {len(df_chunk)} lignes à traiter.")

        last_entry_date, last_entry_title = get_last_entry(session)
        data_to_insert = []
        skipped_entries = 0

        for _, row in df_chunk.iterrows():
            try:
                row_date_commentaire = pd.to_datetime(row['date_commentaire'])
                row_titre_com = str(row.get('titre_com', ''))

                # Vérification si la donnée existe déjà
                existing_entry = session.query(CleanedData).filter(
                    CleanedData.date_commentaire == row_date_commentaire,
                    CleanedData.titre_com == row_titre_com
                ).first()

                if existing_entry:
                    skipped_entries += 1
                    continue

                # Vérification si la donnée est plus récente que la dernière entrée
                if last_entry_date and row_date_commentaire <= last_entry_date:
                    skipped_entries += 1
                    continue

                row_data = CleanedData(
                    categorie_bis=str(row.get('categorie_bis', '')),
                    companies=str(row.get('companies', '')),
                    noms=str(row.get('noms', '')),
                    titre_com=row_titre_com,
                    commentaire=str(row.get('commentaire', '')),
                    reponses=str(row.get('reponses', '')),
                    notes=pd.to_numeric(row.get('notes', 0), errors='coerce'),
                    date_experience=str(row.get('date_experience', '')),
                    date_commentaire=row_date_commentaire,
                    site=str(row.get('site', '')),
                    nombre_pages=pd.to_numeric(row.get('nombre_pages', 0), errors='coerce'),
                    date_scrap=str(row.get('date_scrap', '')),
                    verified=bool(row.get('verified', False)),
                    année_experience=pd.to_numeric(row.get('année_experience', 0), errors='coerce'),
                    mois_experience=pd.to_numeric(row.get('mois_experience', 0), errors='coerce'),
                    jour_experience=pd.to_numeric(row.get('jour_experience', 0), errors='coerce'),
                    année_commentaire=pd.to_numeric(row.get('année_commentaire', 0), errors='coerce'),
                    mois_commentaire=pd.to_numeric(row.get('mois_commentaire', 0), errors='coerce'),
                    jour_commentaire=pd.to_numeric(row.get('jour_commentaire', 0), errors='coerce'),
                    leadtime_com_exp=pd.to_numeric(row.get('leadtime_com_exp', 0), errors='coerce'),
                    nombre_caractères=pd.to_numeric(row.get('nombre_caractères', 0), errors='coerce'),
                    nombre_maj=pd.to_numeric(row.get('nombre_maj', 0), errors='coerce'),
                    nombre_car_spé=pd.to_numeric(row.get('nombre_car_spé', 0), errors='coerce'),
                    caractères_spé=str(row.get('caractères_spé', '')),
                    emojis_positifs_count=pd.to_numeric(row.get('emojis_positifs_count', 0), errors='coerce'),
                    emojis_negatifs_count=pd.to_numeric(row.get('emojis_negatifs_count', 0), errors='coerce'),
                    commentaire_text=str(row.get('commentaire_text', '')),
                    langue_bis=str(row.get('langue_bis', '')),
                    last_entry_date=datetime.utcnow()
                )

                data_to_insert.append(row_data)

            except Exception as row_error:
                print(f"[Thread {thread_id}] Erreur lors du traitement d'une ligne: {row_error}")

        print(f"[Thread {thread_id}] Nombre d'éléments à insérer : {len(data_to_insert)}")

        if data_to_insert:
            try:
                session.bulk_save_objects(data_to_insert)
                session.commit()
                print(f"[Thread {thread_id}] Insertion réussie de {len(data_to_insert)} lignes.")
            except SQLAlchemyError as db_error:
                session.rollback()
                print(f"[Thread {thread_id}] Erreur lors de l'insertion en base: {db_error}")
                print(f"Détails de l'erreur: {str(db_error.__dict__)}")
        else:
            print(f"[Thread {thread_id}] Aucune nouvelle donnée à insérer.")

        print(f"[Thread {thread_id}] Traitement terminé. {skipped_entries} entrées ignorées.")
        pbar.update(len(df_chunk))

    except Exception as e:
        print(f"[Thread {thread_id}] Erreur générale: {e}")
    finally:
        session.close()


async def clean_data() -> List[CleanedData]:
    """Fonction pour nettoyer les données et mettre à jour la base de données."""
    try:
        # Appelle la première fonction de nettoyage
        f_data_clean()
        
        # Appelle la deuxième fonction et stocke le résultat
        cleaned_data = f_data_clean_2()
        
        # Vérifier si le DataFrame est vide
        if cleaned_data.empty:
            print("Aucune donnée à nettoyer.")
            return []

        # Exécuter la mise à jour de la base de données
        await run_database(cleaned_data)

        # Convertir les données nettoyées en liste d'objets CleanedData
        cleaned_items = [CleanedData(**item) for item in cleaned_data.to_dict(orient="records")]
        
        print(f"Nombre d'éléments nettoyés : {len(cleaned_items)}")
        return cleaned_items

    except Exception as e:
        print(f"Erreur lors du nettoyage des données : {e}")
        raise

async def run_database(cleaned_data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, update_database_multithread, cleaned_data)


def update_database_multithread(df):
    """Gère l'insertion des données en utilisant plusieurs threads avec barre de progression."""
    session_factory = scoped_session(SessionLocal)

    num_threads = min(9, len(df))  # Ajuster le nombre de threads en fonction de la taille des données
    chunks = np.array_split(df, num_threads)
    
    total_rows = len(df)
    threads = []

    with tqdm(total=total_rows, desc="Insertion des données", unit=" ligne") as pbar:
        for i, chunk in enumerate(chunks):
            t = threading.Thread(target=insert_data_thread, args=(chunk, session_factory, total_rows, pbar, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    session_factory.remove()
