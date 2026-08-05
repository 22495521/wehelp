import csv
import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUT_DIR = Path(__file__).parent / "beforeClean"
OUT_DIR.mkdir(exist_ok=True)

BOARDS = [
    "Baseball",
    "Boy-Girl",
    "C_Chat",
    "HatePolitics",
    "Lifeismoney",
    "Military",
    "PC_Shopping",
    "Stock",
    "Tech_Job",
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(headers)

TARGET = 100000

for board in BOARDS:
    total = 0
    page = 1

    with open(OUT_DIR / f"{board}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        while total < TARGET:
            url = f"https://www.ptt.cc/bbs/{board}/index{page}.html"

            try:
                resp = session.get(url, timeout=10)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
            except requests.exceptions.RequestException:
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            rows = [(a.get_text(strip=True), board) for a in soup.select("div.title a")]

            if not rows:
                break

            need = TARGET - total
            writer.writerows(rows[:need])
            f.flush()
            total += min(len(rows), need)

            page += 1

    print(f"{board} 完成，{total} 筆 → {board}.csv")

print("全部完成")