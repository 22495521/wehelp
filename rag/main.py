from ollama import chat

while True:
    user_input = input("你: ").strip()
    if user_input in ("/bye", "exit", "quit"):
        break
    if not user_input:
        continue

    print("AI: ", end="", flush=True)
    for chunk in chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": user_input}],
        stream=True,
    ):
        print(chunk["message"]["content"], end="", flush=True)
    print()