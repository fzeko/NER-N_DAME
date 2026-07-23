import json
import random
import spacy
from spacy.tokens import DocBin



JSON_DIR = r"d:\Annotation\JSON\Roberta_C1_Correct.json"
MODEL_DIR = r"D:\Modeles_NER_Custom\V2\roberta\model-best"
OUTPUT_DIR = r"D:\Modeles_NER_Custom\V3_solo"
TRAIN = r"D:\Modeles_NER_Custom\V3_solo\corpus_spacy_en\train.spacy"
DEV = r"D:\Modeles_NER_Custom\V3_solo\corpus_spacy_en\dev.spacy"
nlp = spacy.load(MODEL_DIR)

# Eviter les chevauchements

def enlever_chevauchements(spans):
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    result = []

    for span in spans:
        if all(span.start >= s.end or span.end <= s.start for s in result):
            result.append(span)
    return result

# Conversion au format spaCy

def conversion_doc(task):
    text = task["data"].get("text")
    if not text:
        return None

    doc = nlp.make_doc(text)

    spans = []

    annotations = task.get("annotations", [])
    if not annotations:
        return None
    
    for ann in annotations:
        for a in ann.get("result", []):
            if a.get("type") != "labels":
                continue

            value = a.get("value", {})
            start = value.get("start", {})
            end = value.get("end", {})
            labels = value.get("labels", [])

            if start is None or end is None or not labels:
                continue

            label = labels[0]

            span = doc.char_span(
                start,
                end,
                label=label,
                alignment_mode="contract",
            )

            if span is not None:
                spans.append(span)
        
    spans = enlever_chevauchements(spans)
    doc.ents = spans
    return doc

with open(JSON_DIR, "r", encoding="utf-8") as f:
    tasks = json.load(f)

docs = []
for task in tasks:
    doc = conversion_doc(task)
    if doc is not None:
        docs.append(doc)

print(f"Docs valides : {len(docs)}")

# Split

random.seed(42)
random.shuffle(docs)

split = int(len(docs) * 0.8)

train_data = docs[:split]
dev_data = docs[split:]

# Export 

def sauv_docbin(docs, path):
    db = DocBin()
    for doc in docs:
        db.add(doc)
    db.to_disk(path)

sauv_docbin(train_data, TRAIN)
sauv_docbin(dev_data, DEV)

