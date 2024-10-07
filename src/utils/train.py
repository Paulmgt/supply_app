import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, Any
import json
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata
import time
import nltk
import re

# Téléchargement des ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Fonction pour appliquer le POS tagging sur un texte
def POStagging(commentt):  
    text = []
    sentences = sent_tokenize(commentt)
    for s in sentences:
        wordsList = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(wordsList)
        tagged = ' '.join(map(lambda X: '_'.join(X), tagged))
        text.append(tagged)
    return ' '.join(text)

# Fonction de prétraitement du texte
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)  # Garde uniquement les lettres et espaces
    text = POStagging(text)  # Applique le POS tagging
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Fonction pour trouver le seuil optimal
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [fbeta_score(y_true, (y_pred_proba > threshold).astype(int), beta=2, average='binary') for threshold in thresholds]
    return thresholds[np.argmax(f1_scores)]

# Fonction de score personnalisé
def custom_scorer(y_true, y_pred_proba):
    if len(y_pred_proba.shape) == 2:
        y_pred_proba = y_pred_proba[:, 1]
    threshold = find_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba > threshold).astype(int)
    return fbeta_score(y_true, y_pred, beta=2, average='binary')

# Nettoyage des valeurs numériques
def clean_numeric(x):
    if isinstance(x, str):
        return ''.join(filter(str.isdigit, x)) or np.nan
    return x

# Fonction pour retirer les caractères spéciaux et les emojis
def remove_special_characters_and_emojis(text):
    try:
        text = ''.join(char for char in text if unicodedata.category(char).startswith(('L', 'N', 'P')))
        text = re.sub(r'[^\w\sàáâäçèéêëìíîïñòóôöùúûüýÿÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜÝŸ.,!?\'"]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Erreur pendant le nettoyage du texte: {text}")
        print(f"Message d'erreur: {str(e)}")
        raise

# Préparation des données
def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    def load_data(file_path: str) -> pd.DataFrame:
        return joblib.load(file_path).reset_index(drop=True)

    df_clean_2 = load_data("src/models/data_clean_lib")
    new_df = load_data("src/models/new_data_lib")

    print("Forme de df_clean_2:", df_clean_2.shape)
    print("Forme de new_df:", new_df.shape)

    # Préparation de df_clean_2
    if 'commentaire_bis' in df_clean_2.columns:
        df_clean_2['commentaire_text'] = df_clean_2['commentaire_bis']
        df_clean_2 = df_clean_2.drop(columns=['commentaire_bis'])
    
    new_df['notes_bis'] = new_df['notes'].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1})

    # Sélection des colonnes nécessaires
    colonnes_à_conserver = [
        'notes_bis', 'titre_com', 'commentaire_text', 'nombre_caractères',
        'nombre_maj', 'nombre_car_spé', 'emojis_positifs_count',
        'emojis_negatifs_count'
    ]

    df_clean_2 = df_clean_2[colonnes_à_conserver]
    new_df = new_df[colonnes_à_conserver]

    print("Colonnes de df_clean_2 après préparation:", df_clean_2.columns)
    print("Colonnes de new_df après préparation:", new_df.columns)

    # Concaténation
    df_combined = pd.concat([df_clean_2, new_df], ignore_index=True)

    print("Forme de df_combined:", df_combined.shape)

    # Nettoyage et préparation finale
    df_filtered = df_combined.drop_duplicates().dropna(subset=['commentaire_text'])

    numeric_cols = ['nombre_caractères', 'nombre_maj', 'nombre_car_spé', 'emojis_positifs_count', 'emojis_negatifs_count']
    
    df_filtered[numeric_cols] = df_filtered[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_filtered['commentaire_text'] = df_filtered['commentaire_text'].astype(str)

    # Vérification des valeurs manquantes
    print("Valeurs manquantes dans df_filtered:")
    print(df_filtered.isnull().sum())

    # Suppression des lignes avec des valeurs manquantes
    df_filtered = df_filtered.dropna()

    encode_y = LabelEncoder()
    y = encode_y.fit_transform(df_filtered["notes_bis"])

    X = df_filtered.drop(columns=['notes_bis'])

    # Vérification des dimensions avant le split
    print("Forme de X:", X.shape)
    print("Forme de y:", y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vérification des dimensions après le split
    print("Forme de x_train:", x_train.shape)
    print("Forme de y_train:", y_train.shape)
    print("Forme de x_test:", x_test.shape)
    print("Forme de y_test:", y_test.shape)

    return x_train, x_test, y_train, y_test, encode_y


def train_model() -> Dict[str, any]:
    log_file_path = "src/features/resultats_train.txt"
    model_save_path = "src/models/multinomial_nb_model.joblib"
    ensemble_model_save_path = "src/models/ensemble_model_lib"  # Nouveau chemin pour le modèle d'ensemble
    correlation_matrix_path = "src/features/correlation_matrix.png"  # Chemin pour la matrice de corrélation

    try:
        with open(log_file_path, "a") as fichier_t:
            print(f"---------------{datetime.now()}--------------", file=fichier_t)

            # Préparation des données
            start_time = time.time()
            x_train, x_test, y_train, y_test, encode_y = prepare_data()
            print(f"Temps de préparation des données : {time.time() - start_time:.2f} secondes", file=fichier_t)
            print(f"X_train size: {x_train.shape}, X_test size: {x_test.shape}", file=fichier_t)
            print(f"Distribution des classes dans y_train: {np.bincount(y_train)}", file=fichier_t)
            print(f"Distribution des classes dans y_test: {np.bincount(y_test)}", file=fichier_t)

            # Engineering des features (ajout longueur_mots et nb_mots)
            for df in [x_train, x_test]:
                df['longueur_mots'] = df['commentaire_text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
                df['nb_mots'] = df['commentaire_text'].apply(lambda x: len(str(x).split()))

            # Sauvegarder la matrice de corrélation
            save_correlation_matrix(x_train, correlation_matrix_path)

            # Pré-traitement des features
            numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()
            categorical_features.remove('commentaire_text')

            scaler = MinMaxScaler()  # Utiliser MinMaxScaler à la place de StandardScaler
            onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            tfidf = TfidfVectorizer(max_features=10000, use_idf=True, smooth_idf=True, ngram_range=(1, 2))

            # Appliquer les transformations
            x_train_num = scaler.fit_transform(x_train[numeric_features])
            x_train_cat = onehot.fit_transform(x_train[categorical_features])
            x_train_text = tfidf.fit_transform(x_train['commentaire_text'])

            x_test_num = scaler.transform(x_test[numeric_features])
            x_test_cat = onehot.transform(x_test[categorical_features])
            x_test_text = tfidf.transform(x_test['commentaire_text'])

            # Combiner toutes les features
            x_train_combined = hstack([x_train_num, x_train_cat, x_train_text])
            x_test_combined = hstack([x_test_num, x_test_cat, x_test_text])

            # Rééchantillonnage (UnderSampler + SMOTE)
            under_sampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
            x_train_under, y_train_under = under_sampler.fit_resample(x_train_combined, y_train)

            smote = SMOTE(sampling_strategy=0.8, random_state=42)
            x_train_resampled, y_train_resampled = smote.fit_resample(x_train_under, y_train_under)

            print(f"Distribution des classes après rééchantillonnage: {np.bincount(y_train_resampled)}", file=fichier_t)

            # Entraîner Naive Bayes (MultinomialNB)
            nb = MultinomialNB()
            nb.fit(x_train_resampled, y_train_resampled)

            # Enregistrer le modèle Naive Bayes et les encodeurs
            joblib.dump(nb, model_save_path)
            joblib.dump(encode_y, "src/models/encode_y_lib")
            joblib.dump(scaler, "src/models/scaler_lib")
            joblib.dump(onehot, "src/models/onehot_encoder_lib")
            joblib.dump(tfidf, "src/models/tfidf_vectorizer_lib")
            print(f"Modèle Naive Bayes sauvegardé sous {model_save_path}", file=fichier_t)

            # ---- AJOUT DU MODELE D'ENSEMBLE (RandomForestClassifier) ----
            ensemble_model = RandomForestClassifier()
            ensemble_model.fit(x_train_resampled, y_train_resampled)

            # Enregistrez le modèle d'ensemble
            joblib.dump(ensemble_model, ensemble_model_save_path)
            print(f"Modèle RandomForestClassifier sauvegardé sous {ensemble_model_save_path}", file=fichier_t)

            # ------------------------------------------------------------

            # Prédictions et évaluation pour Naive Bayes (même logique que votre code actuel)
            y_pred_proba = nb.predict_proba(x_test_combined)[:, 1]

            # Seuil optimal et évaluation
            def find_optimal_threshold(y_true, y_pred_proba):
                thresholds = np.linspace(0, 1, 100)
                f1_scores = [fbeta_score(y_true, (y_pred_proba > threshold).astype(int), beta=1, average='weighted') for threshold in thresholds]
                return thresholds[np.argmax(f1_scores)]

            best_threshold = find_optimal_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba > best_threshold).astype(int)

            # Enregistrer le seuil optimal
            joblib.dump(best_threshold, "src/models/best_threshold_lib")

            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            # Enregistrer les métriques
            print(f"ROC AUC: {roc_auc:.4f}", file=fichier_t)
            print(f"PR AUC: {pr_auc:.4f}", file=fichier_t)
            print(f"Confusion Matrix:\n{conf_matrix}", file=fichier_t)
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}", file=fichier_t)
            
            # Enregistrement dans le fichier JSON pour l'API
            results = {
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc,
                'Confusion_Matrix': conf_matrix.tolist(),  # Conversion en liste pour JSON
                'Classification_Report': report
            }

            # Retour des résultats
            return {
                'message': 'Modèle entraîné avec succès.',
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': report,
                'best_threshold': best_threshold
            }

    except Exception as e:
        with open(log_file_path, "a") as fichier_t:
            print(f"Erreur lors de l'entraînement du modèle : {str(e)}", file=fichier_t)
        raise



def save_correlation_matrix(data: pd.DataFrame, file_path: str) -> None:
    try:
        # Sélectionner uniquement les colonnes numériques
        numeric_data = data.select_dtypes(include=[np.number])
        # Vérifier s'il y a des colonnes numériques
        if numeric_data.empty:
            raise ValueError("Aucune colonne numérique disponible pour générer une matrice de corrélation.")
        
        # Calculer la matrice de corrélation
        corr_matrix = numeric_data.corr()
        # Générer et sauvegarder la heatmap de corrélation
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matrice de corrélation des features numériques")
        plt.savefig(file_path)
        plt.close()
        print(f"Matrice de corrélation sauvegardée à {file_path}")
    except Exception as e:
        print(f"Erreur lors de la génération de la matrice de corrélation: {str(e)}")



def predict_comment(comment: str, use_naive_bayes=False, apply_pos_tagging=False) -> Dict[str, Any]:
    try:
        # Charger les éléments nécessaires
        model_path_nb = "src/models/multinomial_nb_model.joblib"
        encode_y_path = "src/models/encode_y_lib"
        scaler_path = "src/models/scaler_lib"
        onehot_path = "src/models/onehot_encoder_lib"
        tfidf_path = "src/models/tfidf_vectorizer_lib"
        best_threshold_path = "src/models/best_threshold_lib"
        
        # Chargement des modèles
        encode_y = joblib.load(encode_y_path)
        best_threshold = joblib.load(best_threshold_path)
        scaler = joblib.load(scaler_path)
        onehot = joblib.load(onehot_path)
        tfidf = joblib.load(tfidf_path)

        # Optionnellement appliquer le POS tagging
        if apply_pos_tagging:
            comment = POStagging(comment)

        # Définir l'ordre correct des caractéristiques
        numeric_features = ['nombre_caractères', 'nombre_maj', 'nombre_car_spé', 'emojis_positifs_count', 'emojis_negatifs_count', 'longueur_mots', 'nb_mots']
        categorical_features = ['titre_com']

        # Préparer le commentaire dans un DataFrame
        comment_df = pd.DataFrame({
            'titre_com': ['default_title'],
            'commentaire_text': [comment],
            'nombre_caractères': [len(comment)],
            'nombre_maj': [sum(1 for c in comment if c.isupper())],
            'nombre_car_spé': [sum(1 for c in comment if not c.isalnum())],
            'longueur_mots': [np.mean([len(word) for word in comment.split() if word]) if comment.split() else 0],
            'nb_mots': [len(comment.split())],
            'emojis_negatifs_count': [0],
            'emojis_positifs_count': [0]
        })

        # Vérification des noms de colonnes avant transformation
        print("Colonnes dans comment_df:", comment_df.columns.tolist())
        print("Noms des caractéristiques attendus:", numeric_features)

        # Appliquer les transformations dans le bon ordre
        x_num = scaler.transform(comment_df[numeric_features])
        x_cat = onehot.transform(comment_df[categorical_features])
        x_text = tfidf.transform(comment_df['commentaire_text'])

        # Combiner les caractéristiques dans le bon ordre
        x_transformed = hstack([x_num, x_cat, x_text])

        # Debugging: Afficher la forme et les types des données transformées
        print("Forme des données transformées:", x_transformed.shape)
        print("Caractéristiques numériques:", x_num)
        print("Caractéristiques catégorielles:", x_cat)
        print("Caractéristiques de texte:", x_text)

        if use_naive_bayes:
            nb_model = joblib.load(model_path_nb)
            prediction_nb = nb_model.predict(x_transformed)
            predicted_label_nb = encode_y.inverse_transform(prediction_nb)[0]

            return {
                "message": "Prédiction avec Naive Bayes effectuée avec succès",
                "prediction": predicted_label_nb,
                "score": None,
                "details_probabilities": {
                    "class_0": None,
                    "class_1": None
                },
                "threshold_used": None
            }
        else:
            ensemble_model = joblib.load("src/models/ensemble_model_lib")
            prediction_proba = ensemble_model.predict_proba(x_transformed)[0]
            prediction = (prediction_proba[1] > best_threshold).astype(int)
            predicted_label = encode_y.inverse_transform([prediction])[0]

            return {
                "message": "Prédiction avec modèle d'ensemble effectuée avec succès",
                "score": round(prediction_proba[1] * 100, 2),
                "prediction": predicted_label,
                "details_probabilities": {
                    "class_0": round(prediction_proba[0], 4),
                    "class_1": round(prediction_proba[1], 4)
                },
                "threshold_used": best_threshold
            }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur détaillée : {error_details}")  # Ajout d'un print pour le débogage
        return {
            "message": f"Erreur lors de la prédiction: {str(e)}",
            "error_details": error_details,
            "prediction": "Erreur",
            "score": 0.0,
            "details_probabilities": {
                "class_0": 0.0,
                "class_1": 0.0
            },
            "threshold_used": 0.0
        }