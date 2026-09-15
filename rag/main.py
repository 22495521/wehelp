import os

# CKIP 模型已經下載過，直接用本機快取，不再連 Hugging Face 檢查（也不會出現 HF_TOKEN 警告）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# 關掉 transformers 的 "Loading weights" 進度條與警告
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()  # 要在 import embeding（載入 CKIP）之前

from gensim.models.doc2vec import Doc2Vec
from ollama import chat

from embeding import D2V_PATH, infer, load_article_contents, tokenize


SYSTEM_PROMPT = """你是台灣交通法規助理，請用繁體中文回答。
只根據使用者訊息中提供的「參考法條」回答；如果參考法條裡找不到答案，就直接說不知道，不要自己編造。"""


#
# 檢索：問題 → 最相近的法條
#
def retrieve(model, contents, question, topn=5):
    # 將問題斷詞
    words = tokenize([question])[0]

    articles = []
    for doc_id, score in infer(model, words, topn=topn):
        text = contents[doc_id]
        articles.append((text, score))
    return articles


#
# RAG，組 prompt：參考法條 + 問題
#
def build_prompt(question, articles):
    prompt = ""
    for i, (text, _) in enumerate(articles, start=1):
        prompt += f"【參考法條 {i}】\n"
        prompt += f"{text}\n\n"

    prompt += f"問題：{question}"
    return prompt


if __name__ == "__main__":
    model = Doc2Vec.load(D2V_PATH)
    contents = load_article_contents()

    while True:
        user_input = input("你: ").strip()
        if user_input in ("/bye", "exit", "quit"):
            break
        if not user_input:
            continue

        # 先到embedding  Model 內檢索 
        articles = retrieve(model, contents, user_input)

        print("AI: ", end="", flush=True)
        for chunk in chat(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(user_input, articles)},
            ],
            options={"num_ctx": 8192},
            stream=True,
        ):
            print(chunk["message"]["content"], end="", flush=True)
        print()
