# eval_step0.py
import csv
import glob
import logging
import os
import random

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

random.seed(42)
np.random.seed(42)

# 
# 讀取斷詞後的語料
# 
TOKENIZED_DIR = os.path.join(os.path.dirname(__file__), "afterTokenize")
seen_titles = set()
labels = []
docs = []
for csv_path in sorted(glob.glob(os.path.join(TOKENIZED_DIR, "*_tokenized.csv"))):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            label, tokens = row[0], row[1:]
            if not tokens:
                continue
            key = (label, tuple(tokens))
            if key in seen_titles:
                continue
            seen_titles.add(key)
            labels.append(label)
            docs.append(tokens)

train_corpus = [TaggedDocument(words=t, tags=[i]) for i, t in enumerate(docs)]

#
# 訓練
# 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "doc2vec.model")

if os.path.exists(MODEL_PATH):
    model = Doc2Vec.load(MODEL_PATH)
else:
    model = Doc2Vec(
        vector_size=40, epochs=100,
        workers=os.cpu_count() or 1, seed=42,
        dm=0, window=5,
    )
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_PATH)

# 
# 測試
# 
random.seed(42)
SAMPLE_N = 1000
sample_ids = random.sample(range(len(train_corpus)), SAMPLE_N)

hit1 = 0
hit2 = 0
for n, doc_id in enumerate(sample_ids):
    inferred = model.infer_vector(train_corpus[doc_id].words, epochs=50)
    sims = model.dv.most_similar([inferred], topn=2)
    top2 = [tag for tag, _ in sims]

    if top2[0] == doc_id:
        hit1 += 1
    if doc_id in top2:
        hit2 += 1

print(f"Self Similarity {hit1 / SAMPLE_N:.3f}")
print(f"Second Self Similarity {hit2 / SAMPLE_N:.3f}")
