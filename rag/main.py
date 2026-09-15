from ollama import chat

resp = chat(model="llama3.1:8b", messages=[
    {"role": "user", "content": "全世界最有名的10個旅遊景點是什麼?"}
])


print(resp["message"]["content"])