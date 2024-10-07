import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re
from datetime import datetime
import os

def f_data_clean():
    """Fonction pour nettoyer les données à partir d'un fichier brut."""
    
    fichier_b = open("src/features/log_clean_b.txt", "a")
    print("------------- début clean data----------------- :", file=fichier_b)

    # Fonction pour extraire la note en utilisant une expression régulière
    def extraire_note(description):
        match = re.search(r'Rated\s+(\d+)', str(description))
        return match.group(1) if match else description

    # Fonction pour remplacer certains mots par une chaîne vide
    def remplacer_mots_par_X(phrase, mots_a_remplacer):
        for mot in mots_a_remplacer:
            phrase = phrase.replace(mot, '')
        return phrase

    # Fonction pour convertir la chaîne en date au format "dd/mm/aaaa"
    def convertir_date(chaine):
        try:
            date_obj = datetime.strptime(chaine, "%b %d, %Y")
            return date_obj.strftime("%d/%m/%Y")
        except ValueError:
            return chaine

    def convertir_date2(chaine):
        try:
            date_obj = datetime.strptime(chaine, "%B %d, %Y")
            return date_obj.strftime("%d/%m/%Y")
        except ValueError:
            return chaine

    # Remplacer 'aaaaa pas de commentaire!' par le titre de commentaire
    def replace_value(cell_value, other_value):
        return other_value if cell_value == 'aaaaa pas de commentaire!' else cell_value

    # Si le commentaire est vide (ou au format date) mettre : aaaaa pas de commentaire!.
    def replace_texte(cell_value):
        try:
            pd.to_datetime(cell_value, format='%B %d, %Y')
            return 'aaaaa pas de commentaire!'
        except ValueError:
            return cell_value

    # Vérification de la présence du fichier brut
    if os.path.exists('src/data/Final_data_scraped_brut_latest.csv'):
        df_brut = pd.read_csv('src/data/Final_data_scraped_brut_latest.csv')
        print("Aperçu des données brutes :", df_brut.head(), file=fichier_b)

        df = df_brut

        # Suppression de la colonne 'Unnamed: 0' si elle existe
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        else:
            print('La colonne Unnamed: 0 est déjà supprimée !', file=fichier_b)

        # Application des fonctions de nettoyage
        df['notes'] = df['notes'].apply(extraire_note)
        df['verified'] = df['date_commentaire'].str.contains('Verified').astype(int)

        # Liste des mots à remplacer
        mots_a_remplacer = ["Reviews", "Date of experience:", '<br/><br/>','<br/></p>','</p>','<br/>','<p class="typography_body-m__xgxZ_ typography_appearance-default__AAY17 styles_message__shHhX" data-service-review-business-reply-text-typography="true">']
        mots_a_remplacer_date = ["Verified", "Updated ", "Invited", " ago", "Merged", "Redirected"]

        # Nettoyage des colonnes
        for nom_colonne in df.columns:
            df[nom_colonne] = df[nom_colonne].astype(str)
            df[nom_colonne] = df[nom_colonne].apply(lambda x: remplacer_mots_par_X(x, mots_a_remplacer))

        df['date_commentaire'] = df['date_commentaire'].apply(lambda x: remplacer_mots_par_X(x, mots_a_remplacer_date))

        # Conversion des dates et nettoyage des colonnes
        df['date_commentaire'] = df['date_commentaire'].apply(lambda x: convertir_date(x))
        df['date_experience'] = df['date_experience'].apply(lambda x: convertir_date2(x))

        # Suppression des lignes contenant des chaînes liées au temps dans 'date_commentaire'
        mask = df['date_commentaire'].str.contains(r'\b(days|day|minutes|minute|hours|hour)\b', case=False)
        df = df[~mask]

        # Conversion des types de colonnes
        df["notes"] = pd.to_numeric(df["notes"], errors='coerce')
        df["verified"] = df["verified"].astype(int)
        df["nombre_pages"] = pd.to_numeric(df["nombre_pages"], errors='coerce')
        df["date_experience"] = pd.to_datetime(df["date_experience"], format='%d/%m/%Y', errors='coerce')
        df["date_commentaire"] = pd.to_datetime(df["date_commentaire"], format='%d/%m/%Y', errors='coerce')

        # Création de nouvelles colonnes basées sur les dates
        df["année_experience"] = df['date_experience'].dt.year
        df["mois_experience"] = df['date_experience'].dt.month
        df["jour_experience"] = df['date_experience'].dt.day
        df["année_commentaire"] = df['date_commentaire'].dt.year
        df["mois_commentaire"] = df['date_commentaire'].dt.month
        df["jour_commentaire"] = df['date_commentaire'].dt.day

        # Calcul de l'écart entre la date d'expérience et la date du commentaire
        df['leadtime_com_exp'] = df['date_commentaire'] - df['date_experience']

        # Remplacement des textes vides et des valeurs spécifiques
        df['commentaire'] = df['commentaire'].apply(replace_texte)
        df['commentaire'] = df.apply(lambda row: replace_value(row['commentaire'], row['titre_com']), axis=1)

        # Génération du nom de fichier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f'src/data/Final_data_scraped_traité_non_traduit_{timestamp}.csv'

        # Enregistrement du DataFrame en CSV avec horodatage
        df.to_csv(csv_file, index=False)

        # Création d'un fichier 'latest.csv' qui sera toujours écrasé par le nouveau
        latest_file = 'src/data/Final_data_scraped_traité_non_traduit_latest.csv'
        df.to_csv(latest_file, index=False)

        print("Le nettoyage des données est terminé: (b_data_clean):", file=fichier_b)
    else:
        print("Le fichier brut n'existe pas !!!", file=fichier_b)
    
    fichier_b.close()

    return

if __name__ == "__main__":
    f_data_clean()