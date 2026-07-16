from providers.gemini_provider import GeminiProvider

gemini = GeminiProvider()

response = gemini.generate(
    "Write one sentence introducing Cabangile AI Video Studio."
)

print(response)
