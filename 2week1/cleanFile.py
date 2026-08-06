import csv
from pathlib import Path

BOARDS = [
    "Baseball", "Boy-Girl", "C_Chat", "HatePolitics", "Lifeismoney",
    "Military", "PC_Shopping", "Stock", "Tech_Job",
]



SRC = Path(__file__).parent / "beforeClean"
OUT = Path(__file__).parent / "afterClean"
PREFIXES = ("re:", "fw:")

def strip_reply(value: str) -> str:
    v = value.strip()
    if v.lower().startswith(PREFIXES):
        end = v.find(":")
        if end != -1:
            return v[end + 1:].strip()
    return v


def strip_tag(value: str) -> str:
    v = value.strip()
    if v.startswith("["):
        end = v.find("]")
        if end != -1:
            return v[end + 1:].strip()
    return v

def clean_title(value: str) -> str:
    v = value.strip()
    # 一直清理值到沒有變化。
    while True:
        before = v
        v = strip_reply(v)
        v = strip_tag(v)
        if v == before:
            break
    return v

def process(board: str):
    src = SRC / f"{board}.csv"
    dst = OUT / f"{board}_cleaned.csv"

    with src.open(newline="", encoding="utf-8-sig") as fin, \
         dst.open("w", newline="", encoding="utf-8-sig") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:

            title = clean_title(row[0].strip().lower())
            if not title:
                continue
            row[0] = title
            writer.writerow(row)


OUT.mkdir(exist_ok=True)
for board in BOARDS:
    process(board)
    print(f"[ok] {board}")