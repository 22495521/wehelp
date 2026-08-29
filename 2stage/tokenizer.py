import csv
import re
from pathlib import Path

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

# BOARDS = [
#     "Baseball", "Boy-Girl", "C_Chat", "HatePolitics", "Lifeismoney",
#     "Military", "PC_Shopping", "Stock", "Tech_Job",
# ]
BOARDS = [
    "Baseball"
]



SRC = Path(__file__).parent / "testAfterClean"
OUT = Path(__file__).parent / "testAfterTokenize"

DROP_TAGS = {
    "P",
    "Caa", "Cab", "Cba", "Cbb",
    "DE", "T",
    "D", "Dfa",
    "Nh",
    "SHI",
    "WHITESPACE",
}

ws_driver = CkipWordSegmenter(model="albert-tiny", device=-1)
pos_driver = CkipPosTagger(model="albert-tiny", device=-1)


# 完全由符號／標點組成的詞（沒有任何中文字或英數字）
PUNCT_ONLY = re.compile(r"^[^\w一-鿿]+$")


def filter_words(sentence_ws, sentence_pos, drop=DROP_TAGS):
    result = []
    for word, tag in zip(sentence_ws, sentence_pos):
        word = word.strip()
        if not word:
            continue
        if tag in drop:
            continue
        if tag.endswith("CATEGORY"):
            continue
        if PUNCT_ONLY.match(word):
            continue
        result.append(word)
    return result


def process(board: str):
    src = SRC / f"{board}_cleaned.csv"
    dst = OUT / f"{board}_tokenized.csv"

    with src.open(newline="", encoding="utf-8-sig") as fin:
        reader = csv.reader(fin)
        rows = [row for row in reader if row]

    titles = [row[0] for row in rows]
    labels = [row[1] for row in rows]

    ws = ws_driver(titles)
    pos = pos_driver(ws)

    with dst.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.writer(fout)
        for label, sentence_ws, sentence_pos in zip(labels, ws, pos):
            words = filter_words(sentence_ws, sentence_pos)
            writer.writerow([label, *words])


def tokenize_titles(titles):
    """給一批標題（清理後的文字），回傳每篇對應的斷詞結果（list[list[str]]）。"""
    ws = ws_driver(titles)
    pos = pos_driver(ws)
    return [filter_words(sentence_ws, sentence_pos) for sentence_ws, sentence_pos in zip(ws, pos)]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for board in BOARDS:
        process(board)
        print(f"[ok] {board}")
