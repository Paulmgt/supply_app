from time import sleep
import re
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from datetime import datetime
import logging
from requests.exceptions import RequestException

# Configurer le logger
logging.basicConfig(filename='src/data/scraping.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_page(url: str) -> bs:
    """Télécharge une page et retourne le contenu BeautifulSoup."""
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        return bs(response.content, "lxml")
    except RequestException as e:
        logging.error(f"Erreur lors de la récupération de la page {url}: {e}")
        return None

def extraire_nombre_etoiles(avis) -> int:
    """Extrait le nombre d'étoiles à partir des images d'étoiles."""
    etoiles = avis.find_all('img', alt=True)
    nombre_etoiles = None  # Par défaut, aucune étoile trouvée
    
    for etoile in etoiles:
        alt_text = etoile.get('alt', '')
        if 'Rated' in alt_text:
            try:
                # On extrait le nombre d'étoiles en supposant que le texte est sous la forme "Rated X"
                nombre_etoiles = int(alt_text.split(' ')[1])
                break  # On suppose qu'une seule image d'étoile contient l'information
            except (ValueError, IndexError):
                # En cas de problème de conversion ou de format inattendu
                nombre_etoiles = None
    
    return nombre_etoiles


def scrape_first_phase(url: str) -> pd.DataFrame:
    """Première phase de scraping pour les colonnes marque, liens_marque, catégorie, reviews, pays."""
    categorie, pays, marque, liens_marque, reviews = [], [], [], [], []
    # liste_liens1 = ['events_entertainment'] 
    liste_liens1 = ['food_beverages_tobacco', 'hobbies_crafts', 'business_services', 'animals_pets', 'home_garden',
                    'legal_services_government', 'money_insurance', 'media_publishing', 'public_local_services', 'beauty_wellbeing',
                    'restaurants_bars', 'shopping_fashion', 'construction_manufactoring', 'health_medical', 'sports', 'education_training', 
                    'utilities', 'travel_vacation', 'electronics_technology', 'home_services', 'vehicles_transportation', 'events_entertainment']
    
    for lien_cc in liste_liens1:
        lien = f"{url}/{lien_cc}?country=FR"
        soup = fetch_page(lien)
        if not soup:
            continue
        
        for X in range(1, 3):
            sleep(0.5)  # Attendre une demi-seconde entre chaque page
            lien2 = f"{url}/{lien_cc}?country=FR&page={X}"
            soup2 = fetch_page(lien2)
            if not soup2:
                continue

            soup_marques = soup2.find_all('div', class_="paper_paper__1PY90 paper_outline__lwsUX card_card__lQWDv card_noPadding__D8PcU styles_wrapper__2JOo2")
            
            for lien_m in soup_marques:
                marque_element = lien_m.find('p', class_='typography_heading-xs__jSwUz typography_appearance-default__AAY17')
                marque.append(marque_element.text if marque_element else 'Non spécifié')

                lien_element = lien_m.find('a', class_='link_internal__7XN06 link_wrapper__5ZJEx styles_linkWrapper__UWs5j')
                liens_marque.append(lien_element.get('href') if lien_element else 'Non spécifié')

                review_element = lien_m.find('p', class_='typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_ratingText__yQ5S7')
                reviews.append(review_element.text if review_element else 'Non spécifié')

                categorie.append(lien_cc)
                pays.append("FR")

    df_liens = pd.DataFrame({
        'marque': marque,
        'liens_marque': liens_marque,
        'categorie': categorie,
        'reviews': reviews,
        'pays': pays
    })

    # Data cleaning
    df_liens['liens_marque'] = df_liens['liens_marque'].str.replace('/review/', '')

    def extraire_chiffres(texte):
        pattern = r'\|\</span>([0-9,]+)'
        match = re.search(pattern, str(texte))
        if match:
            return match.group(1)
        elif len(str(texte)) < 8:
            return texte
        else:
            return None

    df_liens['reviews'] = df_liens['reviews'].apply(extraire_chiffres)
    df_liens['reviews'] = df_liens['reviews'].str.replace(',', '').astype(float)

    df_liens = df_liens.sort_values(by=['categorie', 'reviews'], ascending=[True, False])

    # Enregistrer le dataframe traité en CSV avec horodatage
    timestamp_liens = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_liens = f'src/data/Avis_trustpilot_liste_liens_{timestamp_liens}.csv'
    df_liens.to_csv(csv_file_liens, index=False)

    # Optionnel : Créer une copie ou un lien symbolique vers un fichier 'latest.csv'
    latest_file_liens = 'src/data/Avis_trustpilot_liste_liens_latest.csv'
    df_liens.to_csv(latest_file_liens, index=False)

    return df_liens

def scrape_second_phase(df_liens_filtré: pd.DataFrame) -> pd.DataFrame:
    """Deuxième phase de scraping pour les détails des avis clients."""
    date_actuelle = datetime.now()

    Data = {
        'categorie_bis': [],
        'companies': [],
        'noms': [],
        'titre_com': [],
        'commentaire': [],
        'reponses': [],
        'notes': [],
        'date_experience': [],
        'date_commentaire': [],
        'site': [],
        'nombre_pages': [],
        'date_scrap': [],
        'verified': [],
        'année_experience': [],
        'mois_experience': [],
        'jour_experience': [],
        'année_commentaire': [],
        'mois_commentaire': [],
        'jour_commentaire': [],
        'leadtime_com_exp': []
    }

    for lien_cat in df_liens_filtré['categorie'].unique():
        df_marque = df_liens_filtré[df_liens_filtré['categorie'] == lien_cat]

        for lien_c in df_marque['liens_marque']:
            url_lien = f'https://www.trustpilot.com/review/{lien_c}?page=1'

            try:
                page = requests.get(url_lien, verify=False)
                soup = bs(page.content, "lxml")
            except Exception as e:
                print(f"Une exception s'est produite pour {lien_c}: {e}")
                continue

            pagination_div = soup.find('div', class_='styles_pagination__6VmQv')
            nb_pages = 1
            if pagination_div:
                page_numbers = pagination_div.find_all('span')
                if page_numbers:
                    last_page_number = page_numbers[-2].get_text() if len(page_numbers) > 1 else '1'
                    nb_pages = int(last_page_number)
            
            for X in range(1, nb_pages + 1):
                lien = f'https://www.trustpilot.com/review/{lien_c}?page={X}'
                page = requests.get(lien, verify=False)
                soup = bs(page.content, "lxml")
                avis_clients = soup.find_all('div', attrs={'class': "styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ"})

                company = None
                try:
                    company_element = soup.find('h1', class_='typography_default__hIMlQ typography_appearance-default__AAY17 title_title__i9V__')
                    if company_element:
                        company = company_element.text.strip()
                except Exception as e:
                    print(f"Erreur lors de la récupération de la société pour {lien_c}: {e}")

                for avis in avis_clients:
                    try:
                        nom_element = avis.find('span', class_='typography_heading-xxs__QKBS8 typography_appearance-default__AAY17')
                        nom = nom_element.text.strip() if nom_element else 'Non spécifié'

                        titre_element = avis.find('h2', class_='typography_heading-s__f7029 typography_appearance-default__AAY17')
                        titre = titre_element.text.strip() if titre_element else 'Non spécifié'

                        commentaire_element = avis.find('p')
                        commentaire = commentaire_element.text.strip() if commentaire_element else 'Non spécifié'

                        reponse_element = avis.find('p', class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17 styles_message__shHhX')
                        reponse = reponse_element.text.strip() if reponse_element else 'Non spécifié'

                        note = extraire_nombre_etoiles(avis)

                        date_experience_element = avis.find('p', class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17')
                        date_experience = date_experience_element.text.strip() if date_experience_element else 'Non spécifié'

                        date_commentaire_element = avis.find('div', class_='styles_reviewHeader__iU9Px')
                        date_commentaire = date_commentaire_element.text.strip() if date_commentaire_element else 'Non spécifié'
                    except Exception as e:
                        print(f"Erreur lors de l'extraction des données d'avis pour {lien_c}: {e}")
                        continue

                    Data['noms'].append(nom)
                    Data['titre_com'].append(titre)
                    Data['commentaire'].append(commentaire)
                    Data['reponses'].append(reponse)
                    Data['notes'].append(note)
                    Data['date_experience'].append(date_experience)
                    Data['date_commentaire'].append(date_commentaire)
                    Data['companies'].append(company)
                    Data['site'].append(lien)
                    Data['nombre_pages'].append(nb_pages)
                    Data['categorie_bis'].append(lien_cat)
                    Data['date_scrap'].append(date_actuelle.strftime("%d-%m-%Y"))
                    Data['verified'].append(None)
                    Data['année_experience'].append(None)
                    Data['mois_experience'].append(None)
                    Data['jour_experience'].append(None)
                    Data['année_commentaire'].append(None)
                    Data['mois_commentaire'].append(None)
                    Data['jour_commentaire'].append(None)
                    Data['leadtime_com_exp'].append(None)

    df = pd.DataFrame(Data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'src/data/Final_data_scraped_brut_{timestamp}.csv'
    df.to_csv(csv_file, index=False)

    latest_file = 'src/data/Final_data_scraped_brut_latest.csv'
    df.to_csv(latest_file, index=False)
    df_loaded = pd.read_csv(latest_file)

    print("------------------------------------------------------------------------")
    print('#### Résultats: données brutes scrapées:', file=open("src/data/texte.txt", "w"))
    print("Voici le Dataframe des données brutes scrapées (données non traitées). \nD'après ce que nous voyons ci-dessus, les données scrapées nécessitent un traitement supplémentaire avec text mining. Nous allons aussi procéder à la création de nouvelles features engineering.")
    print(df_loaded['categorie_bis'].value_counts())
    print(f"La taille du df brut: {df_loaded.shape}")
    print(f"Webscraping terminé le: {date_actuelle}", file=open("src/data/texte.txt", "w"))
    print(f"Webscraping terminé le: {date_actuelle}")

    return df


if __name__ == "__main__":
    df_lien = scrape_first_phase("https://www.trustpilot.com/categories")
    scrape_second_phase(df_lien)