import requests

def chat_with_ollama(messages):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.2:1b",
        "stream": False,
        "messages": messages
    }

    response = requests.post(url, json=payload)
    return response.json().get("message").get("content")

if __name__ == "__main__":
    # Example usage
    messages = [
        {"role": "user", "content": "why is the sky blue?"}
    ]
    response = chat_with_ollama(messages)
    print(response)