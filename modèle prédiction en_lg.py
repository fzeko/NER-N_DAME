import spacy
import pandas as pd
import random
import matplotlib.pyplot as plt
from spacy.training import Example
from sklearn.metrics import classification_report, precision_recall_fscore_support

# =======================
# Charger et Préparer les données pour l'entrainement du modèle SpaCy
# =======================

dico = pd.read_csv(r"D:/Dictionnaire/EN_FR_Dictionnaire.csv", 
                   encoding="utf-8-sig", 
                   delimiter=";")

# Génération de phrases simples pour chaque entité du dictionnaire, pour obtenir les 
# positions de début et de fin de l'entité dans le texte, ainsi que son label (catégorie).

train_data = []
train_losses = []
#train_f1_scores = []
#train_precision_scores = []
#train_recall_scores = []
epochs = 30

for _, row in dico.iterrows():  # itère sur chaque ligne du DataFrame
    entity = str(row["Entité"])
    label = str(row["Catégorie"])
    
    texts = [
        f"{entity} is mentioned in this document."
    ]  # Génère une phrase simple contenant l'entité
    for text in texts:
        start = text.find(entity) # Trouve la position de début de l'entité
        end = start + len(entity) # Trouve la position de fin de l'entité

        annotations = {"entities": [(start, end, label)]}
        train_data.append((text, annotations))

print(train_data[:3])  # Affiche les trois premières phrases annotées pour vérification

# Configuration du modèle SpaCy pour l'entraînement du NER

nlp = spacy.load("en_core_web_lg") 

ner = nlp.get_pipe("ner") # Récupère le composant NER du pipeline SpaCy pour l'entraînement

# Ajouter les labels
for _, row in dico.iterrows():
    ner.add_label(row["Catégorie"])

# Désactiver les autres composants pour accélérer l'entraînement
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

def evaluate_model(nlp, data):
    examples = []
    
    for text, annotations in data:
        doc = nlp(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
        
    scores = nlp.evaluate(examples)
    return scores

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    
    for epoch in range(epochs): #
        random.shuffle(train_data)
        losses = {} # Dictionnaire pour suivre les pertes d'entraînement
        #f1_scores = {} # Dictionnaire pour suivre les scores F1 d'entraînement

        #loss_train = losses["ner"]
        #loss_val = 0.50
        
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations) 
            nlp.update([example], losses=losses) 
        
        scores = evaluate_model(nlp, train_data)  # Évalue le modèle après chaque époque
        train_losses.append(losses["ner"])
        #train_f1_scores.append(scores['ents_f'])
        #train_precision_scores.append(scores['ents_p'])
        #train_recall_scores.append(scores['ents_r'])

        print(f"Epoch {epoch + 1}")
        print(f"Losses: {losses['ner']:.2f}")  # Affiche les pertes d'entraînement pour chaque époque
        print(f"Précision: {scores['ents_p']:.3f}") #p pour precision, r pour recall et f pour f1-score
        print(f"Recall: {scores['ents_r']:.3f}")
        print(f"F1-score: {scores['ents_f']:.3f}")
        print("-" * 30)

        

plt.figure(figsize=(12, 6))
plt.plot(range(epochs), train_losses, label="Loss d'entraînement", color="blue")
#plt.plot(range(epochs), train_f1_scores, label="F1-score", color="green")
#plt.plot(range(epochs), train_precision_scores, label="Précision", color="orange")
#plt.plot(range(epochs), train_recall_scores, label="Recall", color="red")
plt.title("Évolution de la fonction coût")
plt.xlabel("Époques")
plt.ylabel("Pertes")
plt.legend()

plt.savefig("learning_curve_en_lg.png")  # Sauvegarde la courbe d'apprentissage
plt.show()

nlp.to_disk(r"D:/Models/ner_custom_en_lg")
print("Modèle sauvegardé avec succès.")

