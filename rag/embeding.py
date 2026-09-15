"""用交通法規條文訓練 Doc2Vec 模型。

流程：讀取法條 → 斷詞 → 訓練 → 存檔（doc2vec.model）
"""

import json
import logging
import os
import random
import re

import numpy as np
from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(__file__)
LAW_PATH = os.path.join(BASE_DIR, "traffic-law.json")
D2V_PATH = os.path.join(BASE_DIR, "doc2vec.model")


#
# 斷詞
#

# 要濾掉的詞性（CKIP 標記）
DROP_TAGS = {
    "P",                          # 介詞：在、對、依
    "Caa", "Cab", "Cba", "Cbb",   # 連接詞：及、或、但
    "DE", "T",                    # 的、之、得、了
    "D", "Dfa",                   # 副詞：應、不、很
    "Nh",                         # 代名詞：其、他
    "SHI",                        # 是
    "WHITESPACE",
}

# 只由標點符號組成的詞（沒有中文字或英數字）
PUNCT_ONLY = re.compile(r"^[^\w一-鿿]+$")

ws_driver = CkipWordSegmenter(model="albert-tiny", device=-1)
pos_driver = CkipPosTagger(model="albert-tiny", device=-1)


def is_useful_word(word, tag):
    if not word:
        return False
    if tag in DROP_TAGS or tag.endswith("CATEGORY"):
        return False
    if PUNCT_ONLY.match(word):
        return False
    return True


def tokenize(texts):
    """一批文字 → 每篇的詞列表（list[list[str]]）"""
    words_per_text = ws_driver(texts, show_progress=False)
    tags_per_text = pos_driver(words_per_text, show_progress=False)

    results = []
    for words, tags in zip(words_per_text, tags_per_text):
        kept = [w.strip() for w, t in zip(words, tags) if is_useful_word(w.strip(), t)]
        results.append(kept)
    return results


#
# 讀取法條
#
def load_article_contents(path=LAW_PATH):
    """回傳 LawArticles 裡每一筆的 ArticleContent（略過已廢止的「（刪除）」條文）"""
    with open(path, encoding="utf-8-sig") as f:
        law = json.load(f)

    contents = []
    for article in law["LawArticles"]:
        text = article["ArticleContent"].strip()
        if text:
            contents.append(text.replace("\r\n", "\n"))
    return contents


#
# 訓練
#
def train(docs):
    corpus = [TaggedDocument(words=words, tags=[i]) for i, words in enumerate(docs)]

    model = Doc2Vec(
        vector_size=40,
        epochs=200,
        dm=0,
        seed=42,
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model


#
# 推論
#
def infer(model, words, topn=10):
    inferred = model.infer_vector(words, epochs=50)
    return model.dv.most_similar([inferred], topn=topn)


#
# 驗證
#
def evaluate(model, docs):
    hit1 = 0
    hit2 = 0
    for doc_id, words in enumerate(docs):
        top2 = [tag for tag, _ in infer(model, words, topn=2)]

        if top2[0] == doc_id:
            hit1 += 1
        if doc_id in top2:
            hit2 += 1

    print(f"Self Similarity {hit1 / len(docs):.3f}")
    print(f"Second Self Similarity {hit2 / len(docs):.3f}")


if __name__ == "__main__":
    # 只有直接執行（訓練／驗證）時才印 INFO log，被 import 時不印
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    contents = load_article_contents()
    docs = tokenize(contents)

    if os.path.exists(D2V_PATH):
        model = Doc2Vec.load(D2V_PATH)
        print(f"載入現有模型 {D2V_PATH}（要重訓請先刪除模型檔）")
    else:
        model = train(docs)
        model.save(D2V_PATH)
        print(f"已訓練 {len(docs)} 筆，模型存到 {D2V_PATH}")

    evaluate(model, docs)
