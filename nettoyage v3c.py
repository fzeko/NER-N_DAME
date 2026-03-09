import pymupdf4llm
import fitz
import re
import os

#Les fichiers sont déjà néttoyés des images
path = r"D:/articles_chantier_ndp/sans images/spec"
output_path = r"D:/articles_chantier_ndp/test ocr/spec"

#Fonction pour extraire le texte
def extraire_contenu_propre(path, output_path, x_threshold=40, min_font_size = 9):

    #Obtenir le nombre de pages
    doc_info = fitz.open(path)
    nb_pages = len(doc_info)
    texte_final = []

    #Commencer par le découpage en blocs
    for page_num in range(1, len(doc_info)):
        page = doc_info[page_num]

        textpage = page.get_text(
            "dict",
            flags = fitz.TEXTFLAGS_TEXT,
        )

        text_blocks = [b for b in textpage["blocks"] if b["type"] == 0]

        text_blocks.sort(key=lambda b: b["bbox"][0])

        columns = []
        current_column = []
        last_x = None

        for block in text_blocks:
            x0 = block["bbox"][0]

            if last_x is None:
                current_column.append(block)
                last_x = x0
            elif abs(x0 - last_x) < x_threshold:
                current_column.append(block)
            else:
                columns.append(current_column)
                current_column = [block]
                last_x = x0

        if current_column:
            columns.append(current_column)

        for col in columns:
            col.sort(key=lambda b: b["bbox"][1])
            for block in col:
                block_text = ""

                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > min_font_size:
                            block_text += span["text"]

                    block_text += " "

                clean_text = " ".join(block_text.split())
                if clean_text:
                    texte_final.append(clean_text)

        texte_final.append("\n")
    
    if isinstance(texte_final, list):
        texte_final = "\n".join(texte_final)  # seulement si c'est une liste

    texte_final = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                         "",
                         texte_final)
    texte_final = re.sub(r"\n\d+\s+[A-Z].{10,100}\n", "\n", texte_final)  # Enlever les appels de notes
    texte_final = re.sub(
        r"(?im)^[\s\*_]*(figures?|fig|table)\s*\d+[:.\-]?\s*.*$",
        "",
        texte_final
    )
    texte_final = re.sub(r"\b\-\b", " ", texte_final) 
    texte_final = re.sub(r"https?://\S+|www\.\S+", " ", texte_final) # Supprimer les URLs
    texte_final = re.sub(r"[^a-zA-ZéèêëœæàâäôöûüïîçÉÊÈËŒÆÂÄÀÛÜÔÖÏÎÇ0123456789\s]", "", texte_final) # Supprimer les caractères spéciaux sauf ceux utiles
    texte_final = re.sub(r"\b(\d{1,3}|\d{5,})\b", " ", texte_final) # Supprimer les chiffres
    texte_final = re.sub(r"\b\w\b", " ", texte_final) # Supprimer les mots d'une seule lettre  
    texte_final = re.sub(r"hal|cidcid.*|doi|colcol.*", " ", texte_final, flags=re.IGNORECASE) # Supprimer les références aux figures et à hal et aux images
    texte_final = re.sub(r"\s+", " ", texte_final)  # remplace les espaces multiples
    texte_final = re.sub(
        r"conclusion|introduction|abstract", 
        "", 
        texte_final, 
        flags=re.IGNORECASE
    )

    #Ecrire le texte propre dans un fichier txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(texte_final)
        
    print(f"Le texte propre a été extrait et enregistré dans {output_path}")


#Lancer la fonction pour tous les fichiers du dossier
for filename in os.listdir(path):
    if filename.endswith(".pdf"):
        full_path = os.path.join(path, filename)
        output_file = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.txt")
        extraire_contenu_propre(full_path, output_file)

print(f"Tous les textes propres ont été extraits et enregistrés dans {output_path}")