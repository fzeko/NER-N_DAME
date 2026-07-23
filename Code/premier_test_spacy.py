import spacy
import os
import re
import fitz
import csv
from spacypdfreader import pdf_reader
from spacy.scorer import Scorer
from spacy.training import Example

#Charger le modèle de langue pré-entrainé en anglais (ici de taille moyenne pour une meilleure précision)
nlp = spacy.load("en_core_web_md")
#Localiser les dossiers source et résultat
dossier_source = r"D:/articles_chantier_ndp/EN"
dossier_result = r"D:/articles_chantier_ndp/medium/résultats EN"


#Traiter tous les fichiers PDF du dossier source et les enregistrer sous forme de fichiers CSV dans le dossier résultat
#Définir une fonction pour nettoyer le texte en supprimant les chiffres, les URLs, les caractères spéciaux, les espaces multiples, les mots d'une seule lettre et les références aux figures et à hal et aux images

def cleantext(text):
    text = re.sub(r"\b\d{1,3}\b", "", text) # Supprimer les chiffres
    text = re.sub(r"https?://\S+|www\.\S+", "", text) # Supprimer les URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Supprimer les caractères spéciaux
    text = re.sub(r"\s+", " ", text) # Supprimer les espaces multiples
    text = re.sub(r"\b\w\b", "", text) # Supprimer les mots d'une seule lettre  
    text = re.sub(r"fig|hal|cidcid.*|doi", "", text, flags=re.IGNORECASE) # Supprimer les références aux figures et à hal et aux images
    return text

for filename in os.listdir(dossier_source):
    if filename.endswith(".pdf"):
        path = os.path.join(dossier_source, filename)
        #Ouvrir le fichier PDF et extraire le texte de chaque page sauf la première page
        with fitz.open(path) as fitz_doc:
                text = ""
                for page in range(1, len(fitz_doc)):
                    text += fitz_doc[page].get_text()

        doc = nlp(text)
        nom = filename.replace(".pdf", ".csv")

        #Ouvrir le fichier CSV en mode écriture et enregistrer les entités extraites avec leur type et leur position dans le texte
        with open(os.path.join(dossier_result, nom), "w", encoding="utf-8", newline = '') as f:
            tableur = csv.writer(f, delimiter='\t')
            tableur.writerow(["Entité", "Type", "Position de début", "Position de fin"])
            
            for ent in doc.ents:
                clean_name = cleantext(ent.text).strip()
                if clean_name:  # Vérifier que le nom nettoyé n'est pas vide
                    tableur.writerow([clean_name, ent.label_, ent.start_char, ent.end_char])
        print (f"Résultats enregistrés dans {dossier_result}")

print("Traitement terminé pour tous les fichiers PDF.")
