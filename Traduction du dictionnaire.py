import pandas as pd
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

#print(df.columns.tolist())

df = pd.read_csv(r"D:\Dictionnaire\Dictionnaire_Mots_Clés_Pierre.csv", 
                 encoding="utf-8-sig", 
                 delimiter=";")

llm = OllamaLLM(model="qwen2.5:7b", temperature=0.2, max_tokens=100)
#qwen2.5:7b
def traduction(entites):
    prompt = f"""
Tu es un traducteur professionnel spécialisé dans la traduction de termes techniques et de mots-clés liés à aux techniques du patrimoine.
Traduis cette liste d'entités en français. 
Réponds uniquement par la traduction,sans numération, ni puces, ni tirets, ni virgules, sans aucune autre forme de présentation.
Garde l'ordre de la liste. Les mots déjà en français ne doivent pas être traduits.
Les éléments chimiques, les formules et les symboles ne doivent pas être traduits.

Suis la logique des exemples suivants :
- "space" devient "espace"
- "tympanums" devient "tympans"
- "interfaces" reste "interfaces"
- "TGA" devient "ATG"
- "scanner" reste "scanner"
- "facing hammer" devient "marteau de façonnage"
- "90s" devient "années 90"


{entites}
    """
    return llm.invoke(prompt).strip()

df["Entité"] = df["Entité"].apply(traduction)

print("Traduction terminée.")

df.to_csv(r"D:\Dictionnaire\Traduction_Mots_Clés.csv", 
          index=False, 
          encoding="utf-8-sig", 
          sep=";")

print("Traduction sauvegardée.")