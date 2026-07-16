import os
import requests
import json

API_KEY = os.environ["GEMINI_API_KEY"]

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Write a one-sentence introduction for Cabangile AI Video Studio."
                }
            ]
        }
    ]
}

response = requests.post(url, json=payload)

print(json.dumps(response.json(), indent=2))
