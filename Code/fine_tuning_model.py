import random
import os
import spacy
import torch  
from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm 

# Paramètres
MODEL_DIR = r"D:\Modeles_NER_Custom\V3_solo\roberta"
TRAIN = r"D:\Modeles_NER_Custom\Test_final\corpus_spacy_en\train.spacy"
DEV = r"D:\Modeles_NER_Custom\Test_final\corpus_spacy_en\dev.spacy"
OUTPUT_DIR = r"D:\Modeles_NER_Custom\Test_final\V3_solo\roberta"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
spacy.require_gpu()
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU : {device_name}")
 
nlp = spacy.load(MODEL_DIR)
#transformer = nlp.get_pipe("transformer")
#print(f"[INFO] Attributs du Transformer (Window/Stride) : {transformer.model.attrs}\n")
transformer = nlp.get_pipe("transformer")
transformer.model.attrs["trainable"] = True

BATCH_SIZE = 2     
EVAL_BATCH_SIZE = 2

# Charger train/dev data
train_db = DocBin().from_disk(TRAIN)
dev_db = DocBin().from_disk(DEV)

train_doc = list(train_db.get_docs(nlp.vocab))
dev_doc = list(dev_db.get_docs(nlp.vocab))

print("Train docs:", len(train_doc))
print("Dev docs:", len(dev_doc))
print("Train entities:", sum(len(doc.ents) for doc in train_doc))
print("Dev entities:", sum(len(doc.ents) for doc in dev_doc))

def doc_example(nlp, docs):
    examples = []
    for doc in docs:
        examples.append(Example(doc, doc))
    return examples

train_ex = doc_example(nlp, train_doc)
dev_ex = doc_example(nlp, dev_doc)

train_texts = {doc.text for doc in train_doc}
dev_texts = {doc.text for doc in dev_doc}


optimizer = nlp.resume_training()
best_f1 = 0.0
print(nlp.config["training"]["optimizer"])

print(f"Lancement de l'entraînement sur {len(train_ex)} documents")
print(nlp.pipe_names)

for epoch in range(10):
    random.shuffle(train_ex)
    losses = {}

    progress_bar = tqdm(
        total=len(train_ex), 
        desc=f"Époque {epoch + 1}/10", 
        unit="doc",
        leave=True
    )

    for i in range(0, len(train_ex), BATCH_SIZE):
        batch = train_ex[i : i + BATCH_SIZE]
        
        nlp.update(
            batch, 
            sgd=optimizer,
            drop=0.4,
            losses=losses,
        )
        
        formatted_losses = {k: f"{v:.4f}" for k, v in losses.items()}
        progress_bar.update(len(batch))
        progress_bar.set_postfix(formatted_losses)
        
 
    progress_bar.close()


    scores = nlp.evaluate(dev_ex, batch_size=EVAL_BATCH_SIZE)

 
    print(f"Époque {epoch + 1}:")
    print(f"Précision : {scores['ents_p']:.3f} | Rappel : {scores['ents_r']:.3f} | Loss : {losses} | F1-score : {scores['ents_f']:.4f}")

    if scores['ents_f'] > best_f1:
        best_f1 = scores['ents_f']
        nlp.to_disk(OUTPUT_DIR)
        print(f"Modèle sauvegardé (F1={best_f1:.3f})")
    
    print("-" * 60)
