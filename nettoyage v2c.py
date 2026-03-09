import pymupdf4llm
import fitz
import re
import os

#Les fichiers sont déjà néttoyés des images
path = r"D:/articles_chantier_ndp/sans images/2c"
output_path = r"D:/articles_chantier_ndp/test ocr/2c"

#Fonction pour extraire le texte
def extraire_contenu_propre(path, output_path):

    #Obtenir le nombre de pages
    doc_info = fitz.open(path)
    nb_pages = len(doc_info)
    texte_final = ""

    #Commencer par le découpage en blocs
    for page_num in range(1, len(doc_info)):
        page = doc_info[page_num]

        #Définir la taille des rectangles et le milieu
        rect = page.rect
        mid_x = rect.x0 + rect.width / 2

        # Définir mes deux zones (droite et gauche)
        zone_gauche = fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1)
        zone_droite = fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1)

        #Extraire le texte de chaque zone, gauche puis droite)
        texte_gauche = page.get_text("text", clip=zone_gauche)
        texte_droite = page.get_text("text", clip=zone_droite)

        texte_final += texte_gauche.strip() + "\n" + texte_droite.strip()

    #Définir un pattern pour repérer les titres de sections de fin d'article
    pattern_sections = re.compile(
        r"(?im)^\s*[*#\s]*(acknowledg(e)?ments?|remerciements|data availability|supplementary materials|supplementary information|bibliograph(y|ie)|references|conflicts? of interest|author contributions?|works cited|literature cited|sources|ref)\b",
        re.IGNORECASE
    )
    match = pattern_sections.search(texte_final)

    lignes = texte_final.splitlines()
    index_coupure = None

    # Si le pattern est trouvé, on coupe
    for i, ligne in enumerate(lignes):
        if pattern_sections.search(ligne):
            index_coupure = i
            break
    
    if index_coupure is not None:
        texte_final = "\n".join(lignes[:index_coupure])

    #Nettoyage
    texte_final = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                         "",
                         texte_final)
    texte_final = re.sub(r"\b\-\b", " ", texte_final) 
    texte_final = re.sub(r"\n\d+\s+[A-Z].{10,100}\n", "\n", texte_final)  # Enlever les appels de notes
    texte_final = re.sub(
        r"(?im)^[\s\*_]*(figures?|fig|table)\s*\d+[:.\-]?\s*.*$",
        "",
        texte_final
    )
    texte_final = re.sub(r"https?://\S+|www\.\S+", " ", texte_final) # Supprimer les URLs
    texte_final = re.sub(r"[^a-zA-ZéèêëœæàâäôöûüïîçÉÊÈËŒÆÂÄÀÛÜÔÖÏÎÇ0123456789\s]", "", texte_final) # Supprimer les caractères spéciaux sauf ceux utiles
    texte_final = re.sub(r"\b(\d{1,3}|\d{5,})\b", " ", texte_final) # Supprimer les chiffres
    texte_final = re.sub(r"\s+", " ", texte_final) # Supprimer les espaces multiples
    texte_final = re.sub(r"\b\w\b", " ", texte_final) # Supprimer les mots d'une seule lettre  
    texte_final = re.sub(r"hal|cidcid.*|doi|colcol.*", " ", texte_final, flags=re.IGNORECASE) # Supprimer les références aux figures et à hal et aux images
    texte_final = re.sub(
        r"conclusion|introduction|abstract", 
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