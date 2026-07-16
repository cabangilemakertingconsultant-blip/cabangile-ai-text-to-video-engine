"""
Cabangile AI Video Studio
Gemini Provider

Uses the Gemini REST API with the requests library.
"""

import os
import requests


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.model = model
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:

        payload = {
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise RuntimeError(
                f"Unexpected Gemini response:\n{data}"
            )

    def generate_video_script(
        self,
        topic: str,
        duration: str = "60 seconds",
    ) -> str:

        prompt = f"""
You are a professional video script writer.

Write a complete AI video script.

Topic:
{topic}

Duration:
{duration}

Format:

Title:

Scene 1:
Narration:

Image Prompt:

Scene 2:
Narration:

Image Prompt:

Scene 3:
Narration:

Image Prompt:

Ending:
"""

        return self.generate(prompt)

    def improve_prompt(self, prompt: str) -> str:

        return self.generate(
            f"""
Improve this AI image prompt while keeping its meaning.

Prompt:

{prompt}
"""
        )
