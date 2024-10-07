from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

class UserIn(BaseModel):
    username: str
    first_name: str
    last_name: str
    password: str
    acces: Literal['user', 'admin', 'superadmin'] = 'user'  # Valeur par défaut pour les nouveaux utilisateurs

class UserUpdate(BaseModel):
    username: str
    new_access: str

class UserOut(BaseModel):
    username: str
    first_name: str
    last_name: str
    acces: str

class Token(BaseModel):
    access_token: str
    token_type: str
    type_acces: str

class Lien(BaseModel):
    lien: str

# Configuration de la base de données avec SQLAlchemy
SQLALCHEMY_DATABASE_URL = "sqlite:///./final_data.db"  # Changez selon votre base de données
# Créer un moteur SQLAlchemy
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # Paramètre spécifique à SQLite
)

# Créer une factory de session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base SQLAlchemy pour le modèle de base de données
Base = declarative_base()

# Modèle de base de données SQLAlchemy
class CleanedDataDB(Base):
    __tablename__ = 'cleaned_data'
    
    id = Column(Integer, primary_key=True, index=True)
    categorie_bis = Column(String)
    companies = Column(String)
    noms = Column(String)
    titre_com = Column(String)
    commentaire = Column(String, unique=True)
    reponses = Column(String)
    notes = Column(Float, nullable=True)
    date_experience = Column(String)
    date_commentaire = Column(String)
    site = Column(String)
    nombre_pages = Column(Integer, nullable=True)
    date_scrap = Column(String)
    verified = Column(Boolean)
    année_experience = Column(Integer, nullable=True)
    mois_experience = Column(Integer, nullable=True)
    jour_experience = Column(Integer, nullable=True)
    année_commentaire = Column(Integer, nullable=True)
    mois_commentaire = Column(Integer, nullable=True)
    jour_commentaire = Column(Integer, nullable=True)
    leadtime_com_exp = Column(Float, nullable=True)
    nombre_caractères = Column(Integer, nullable=True)
    nombre_maj = Column(Integer, nullable=True)
    nombre_car_spé = Column(Integer, nullable=True)
    caractères_spé = Column(String)
    emojis_positifs_count = Column(Integer, nullable=True)
    emojis_negatifs_count = Column(Integer, nullable=True)
    commentaire_text = Column(String)
    langue_bis = Column(String)
    last_entry_date = Column(DateTime, default=datetime.utcnow)

# Modèle Pydantic pour la validation des données
class CleanedData(BaseModel):
    categorie_bis: str
    companies: str
    noms: str
    titre_com: str
    commentaire: str
    reponses: str
    notes: Optional[float] = None
    date_experience: str
    date_commentaire: str
    site: str
    nombre_pages: Optional[int] = None
    date_scrap: str
    verified: bool
    année_experience: Optional[int] = None
    mois_experience: Optional[int] = None
    jour_experience: Optional[int] = None
    année_commentaire: Optional[int] = None
    mois_commentaire: Optional[int] = None
    jour_commentaire: Optional[int] = None
    leadtime_com_exp: Optional[float] = None
    nombre_caractères: Optional[int] = None
    nombre_maj: Optional[int] = None
    nombre_car_spé: Optional[int] = None
    caractères_spé: str
    emojis_positifs_count: Optional[int] = None
    emojis_negatifs_count: Optional[int] = None
    commentaire_text: str
    langue_bis: str

    class Config:
        arbitrary_types_allowed = True

# Créer les tables dans la base de données
Base.metadata.create_all(bind=engine)

class CleanResponse(BaseModel):
    message: str
    sample: List[CleanedData]

class Train(BaseModel):
    train: str
    
class PredictRequest(BaseModel):
    comment: str

class PredictResponse(BaseModel):
    message: str
    score: float
    prediction: str
    details_probabilities: Dict[str, float]
    threshold_used: float

class ScraperResponse(BaseModel):
    message: str
    lien: str
    sample: List[Dict[str, Any]]