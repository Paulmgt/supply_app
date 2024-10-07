from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
from src.users.models import UserIn, UserOut, Token, UserUpdate, ScraperResponse, CleanResponse, PredictRequest, PredictResponse
from src.clean.scrapper import scrape_first_phase, scrape_second_phase
import cProfile
import pstats
import io
from fastapi.security import OAuth2PasswordRequestForm
from src.utils.train import train_model, predict_comment
from src.utils.utils import (
    create_access_token, 
    get_current_user, 
    verify_password, 
    load_users, 
    save_user,
    clean_data,
    hash_password
)
from datetime import timedelta

# Router pour les routes utilisateur
user_routes = APIRouter(tags=['user'])
admin_routes = APIRouter(tags=['admin'])

@user_routes.get("/users", response_model=List[UserOut])
async def get_users(token: str = Depends(get_current_user)):
    current_user = get_current_user(token)
    if current_user.acces != 'superadmin':
        raise HTTPException(status_code=403, detail="Accès non autorisé")
    users = load_users()
    return [UserOut(**user.dict()) for user in users]

@admin_routes.post("/update_access")
async def update_access(user_update: UserUpdate, token: str = Depends(get_current_user)):
    current_user = get_current_user(token)
    if current_user.acces != 'superadmin':
        raise HTTPException(status_code=403, detail="Accès non autorisé")

    users = load_users()
    user_found = False

    for user in users:
        if user.username == user_update.username:
            user.acces = user_update.new_access
            user_found = True
            break

    if not user_found:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    save_user(user)
    return {"message": "Accès mis à jour avec succès"}

@user_routes.post("/register", response_model=UserOut)
async def register(user: UserIn):
    """Enregistrement des utilisateurs"""
    users = load_users()
    if not users:
        user.acces = 'superadmin'
    else:
        user.acces = 'user'

    hashed_password = hash_password(user.password)
    user_data = user.dict(exclude={"password"})
    user_in_db = UserIn(**user_data, password=hashed_password)
    save_user(user_in_db)
    return UserOut(**user_data)

@user_routes.post("/logout")
async def logout(current_user: UserOut = Depends(get_current_user)):
    """Déconnexion de l'utilisateur"""
    # Ici, vous pouvez ajouter la logique de gestion de session si nécessaire
    return {"message": "Déconnexion réussie"}


@user_routes.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Obtenir un token d'accès"""
    users = load_users()
    for user in users:
        if user.username == form_data.username and verify_password(form_data.password, user.password):
            token_data = {"sub": user.username}
            access_token = create_access_token(token_data, timedelta(minutes=30))
            return {"access_token": access_token, "token_type": "bearer", "type_acces": user.acces}
    raise HTTPException(status_code=400, detail="Invalid credentials")


@admin_routes.post("/scraper", response_model=ScraperResponse)
async def scraper(current_user: UserOut = Depends(get_current_user)):
    """Scraper les données (réservé aux admins)"""
    if current_user.acces in ["admin", "superadmin"]:
        try:
            # Étape 1 : Scraping des données de la première phase
            url = "https://www.trustpilot.com/categories"
            df_liens = scrape_first_phase(url)
            
            # Étape 2 : Scraping des avis clients avec les données obtenues de la première phase
            df_sample_data = scrape_second_phase(df_liens)
            
            # Convertir le DataFrame en liste de dictionnaires
            sample_data = df_sample_data.to_dict(orient='records') if not df_sample_data.empty else []
            
            # Log des données brutes pour débogage
            print("Données de scraping : ", json.dumps(sample_data, indent=2))
            
            # Vérifiez le format des données
            expected_keys = [
                "categorie_bis", "companies", "noms", "titre_com", "commentaire", "reponses", "notes", 
                "date_experience", "date_commentaire", "site", "nombre_pages", "date_scrap", 
                "verified", "année_experience", "mois_experience", "jour_experience", 
                "année_commentaire", "mois_commentaire", "jour_commentaire", "leadtime_com_exp"
            ]
            for item in sample_data:
                if not isinstance(item, dict):
                    raise ValueError("Les données de scraping ne sont pas des dictionnaires")
                missing_keys = [key for key in expected_keys if key not in item]
                if missing_keys:
                    raise ValueError(f"Les données de scraping sont manquantes pour les clés: {missing_keys}")

            return ScraperResponse(
                message="Scraping terminé pour https://www.trustpilot.com/categories",
                lien="https://www.trustpilot.com/categories",
                sample=sample_data
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur lors du scraping : {str(e)}")
    raise HTTPException(status_code=403, detail="Accès interdit")


@admin_routes.post("/clean", response_model=CleanResponse)
async def clean(current_user: UserOut = Depends(get_current_user)):
    """Nettoyage des données (réservé aux superadmin)"""
    if current_user.acces != "superadmin":
        raise HTTPException(status_code=403, detail="Accès interdit")

    try:
        cleaned_data = await clean_data()
        if not cleaned_data:
            raise HTTPException(status_code=404, detail="Aucune donnée nettoyée disponible")
        
        return {"message": "Nettoyage terminé", "sample": cleaned_data[:5]}  # Retourner seulement les 5 premiers éléments comme échantillon
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


@admin_routes.post("/train", response_model=Dict[str, Any], name="Entraîner les données")
async def train(current_user: UserOut = Depends(get_current_user)):
    """Entraînement des données avec des algorithmes et création de pipelines."""
    
    # Création du token d'accès si nécessaire
    token_data = {"sub": current_user.username}
    access_token = create_access_token(token_data, expires_delta=timedelta(minutes=30))

    # Chargement des utilisateurs et vérification des droits d'accès
    users = load_users()
    user = next((user for user in users if user.username == current_user.username), None)

    if not user or user.acces != "superadmin":
        raise HTTPException(status_code=403, detail="Accès interdit")

    try:
        # Initialisation du profiler
        profiler = cProfile.Profile()
        profiler.enable()

        # Appel de la fonction d'entraînement du modèle
        resultats = train_model()

        # Arrêt du profiler
        profiler.disable()

        # Capturer les résultats du profiler
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Afficher les 20 premières lignes de profiling

        # Récupération des résultats du profiler
        profiler_results = s.getvalue()

        # Journalisation de l'entraînement et des résultats
        with open("src/features/log_app_api.txt", "a") as fichier:
            print("Entraînement des données terminé avec succès.", file=fichier)
            print(f"Utilisateur: {current_user.username} Date: {datetime.now()}", file=fichier)
            print("Résultats du profiler:", profiler_results, file=fichier)

        # Retour des résultats sous forme de JSON
        return {
            "access_token_private": access_token,  # Optionnel, à retirer si non utilisé
            "token_type": "bearer",
            "confusion_matrix": resultats.get("confusion_matrix", []),
            "score": resultats.get("score", 0),
            "classification_report": resultats.get("classification_report", {}),
            "profiler_results": profiler_results  # Résultats du profiling
        }

    except HTTPException as e:
        # Relancer l'exception HTTP si nécessaire (fichier non trouvé, accès interdit)
        raise e
    except Exception as e:
        # Gestion des erreurs internes
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

from fastapi import HTTPException, Depends
from pydantic import BaseModel
from typing import Any, Dict

# Exemple des modèles Pydantic
class PredictRequest(BaseModel):
    comment: str

class PredictResponse(BaseModel):
    message: str
    prediction: str
    score: float
    details_probabilities: Dict[str, float]
    threshold_used: float


@user_routes.post("/predict", response_model=PredictResponse, name="Prédiction")
async def predict(
    request: PredictRequest, 
    current_user: UserOut = Depends(get_current_user)
) -> PredictResponse:
    """Effectue une prédiction en fonction du type d'accès de l'utilisateur"""

    type_acces = current_user.acces

    # Vérifier le type d'accès de l'utilisateur
    if type_acces not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Type d'accès non valide.")

    try:
        # Effectuer la prédiction
        resultats = predict_comment(request.comment)

        # Validation des champs attendus dans les résultats
        expected_keys = ["message", "prediction", "score", "details_probabilities", "threshold_used"]
        
        if not all(key in resultats for key in expected_keys):
            raise HTTPException(status_code=500, detail="Résultats de la prédiction incomplets.")

        # Vérification des types (exemple: conversion de la prédiction en chaîne si nécessaire)
        prediction_str = str(resultats["prediction"])  # Si `prediction` est un entier, on le convertit en chaîne

        # Retourner les résultats dans le format attendu par PredictResponse
        return PredictResponse(
            message=resultats["message"],
            prediction=prediction_str,
            score=float(resultats["score"]),  # Assurez-vous que c'est un float
            details_probabilities=resultats["details_probabilities"],  # Doit être un dict
            threshold_used=float(resultats["threshold_used"])  # Assurez-vous que c'est un float
        )

    except HTTPException as e:
        raise e  # Si c'est une HTTPException, on la relance telle quelle
    except Exception as e:
        # Gestion des erreurs internes, renvoyer une erreur générique en production
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

