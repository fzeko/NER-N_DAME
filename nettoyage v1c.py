import pymupdf4llm
import fitz
import re
import os

#Les fichiers sont déjà néttoyés des images
path = r"D:/articles_chantier_ndp/sans images/1c"
output_path = r"D:/articles_chantier_ndp/test ocr/1c"

#Fonction pour nettoyer le texte
def extraire_contenu_propre(path, output_path):

    #Obtenir le nombre de pages
    doc_info = fitz.open(path)
    nb_pages = len(doc_info)

    #Extraire toutes pages sauf la première
    if nb_pages > 1:
        page_extrait = list(range(1, nb_pages))
    else:
        page_extrait = [0]

    md_text = pymupdf4llm.to_markdown(path, pages=page_extrait) 

    #Définir un pattern pour repérer les titres de sections de fin d'article
    pattern_sections = re.compile(
        r"^\s*[*#\s]*(acknowledg(e)?ments?|remerciements|data availability|copyright|supplementary materials|supplementary information|bibliograph(y|ie)|references|fundings?|Funding|conflicts? of interest|author contributions?|works cited|literature cited|sources|ref)\b",
        re.IGNORECASE | re.MULTILINE
    )

    match = pattern_sections.search(md_text)

    if match:
        md_text = md_text[:match.start()]

    lignes = md_text.splitlines()
    index_coupure = len(lignes)  # Par défaut, on garde tout

    index_coupure = None


    for i, ligne in enumerate(lignes):
        ligne_clean = ligne.strip()
        if pattern_sections.match(ligne_clean):
            index_coupure = i
            break

    if index_coupure is None:
        for i, ligne in enumerate(lignes):
            if i > len(lignes) * 0.76:
                if re.match(r"^\d{3,4}\s+[A-Z]", ligne.strip()):
                    index_coupure = i
                    break

    if index_coupure is None:
        index_coupure = len(lignes)  # Par défaut, on garde tout

    # Nettoyer le texte extrait
    texte_final = "\n".join(lignes[:index_coupure])
    texte_final = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                         "",
                         texte_final)
    texte_final = re.sub(r"\n\d+\s+[A-Z].{10,100}\n", "\n", texte_final)  # Enlever les appels de notes
    texte_final = re.sub(
        r"(?im)^[\s\*_]*(figures?|fig|table)\s*\d+[:.\-]?\s*.*$",
        "",
        texte_final
    )
    texte_final = re.sub(r"\t", " ", texte_final)   # remplace tabulations
    texte_final = re.sub(r"\b\-\b", " ", texte_final) 
    texte_final = re.sub(r"\s*\d+\s*$", " ", texte_final, flags=re.MULTILINE)
    texte_final = re.sub(r"https?://\S+|www\.\S+", " ", texte_final) # Supprimer les URLs
    texte_final = re.sub(r"[^a-zA-ZéèêëœæàâäôöûüïîçÉÊÈËŒÆÂÄÀÛÜÔÖÏÎÇ0123456789\s]",
                        "", 
                        texte_final,
                        flags=re.MULTILINE
                        ) # Supprimer les caractères spéciaux sauf ceux utiles 
    texte_final = re.sub(r"\b\d{1,3}\b", " ", texte_final) # Supprimer les chiffres
    texte_final = re.sub(r"^\s*\d+\s*$", "", texte_final, flags=re.MULTILINE)
    texte_final = re.sub(r"\d*\.\d*", "", texte_final)
    texte_final = re.sub(r"\b\d{5,}\b", " ", texte_final)
    texte_final = re.sub(r"\s+", " ", texte_final) # Supprimer les espaces multiples
    texte_final = re.sub(r"\b\w\b", " ", texte_final) # Supprimer les mots d'une seule lettre  
    texte_final = re.sub(r"hal|cidcid.*|doi|colcol.*", 
                        " ", 
                        texte_final, 
                        flags=re.IGNORECASE
                        ) # Supprimer les références aux figures et à hal et aux images
    texte_final = re.sub(
        r"acknowledg(e)?ments?.*",
        "",
        texte_final,
        flags=re.IGNORECASE | re.DOTALL
    )
    texte_final = re.sub(
        r"conclusion|introduiction|abstract", 
        "", 
        texte_final, 
        flags=re.IGNORECASE
    )
    
    #Ecrire le texte propre dans un fichier txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texte_final.strip())
        
    print(f"Le texte propre a été extrait et enregistré dans {output_path}")


#Lancer la fonction pour tous les fichiers du dossier
for filename in os.listdir(path):
    if filename.endswith(".pdf"):
        full_path = os.path.join(path, filename)
        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.txt")
        extraire_contenu_propre(full_path, output_file)

print(f"Tous les textes propres ont été extraits et enregistrés dans {output_path}")