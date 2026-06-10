from google import genai
import os

# Access the Gemini API key
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Load analyze tests logs prompt
with open("prompts/analyze-tests-logs.txt") as f:
    prompt = f.read()

# Load tests logs
with open("tests_logs.txt") as f:
    tests_logs = f.read()

# Send the request to the Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""
    {prompt}
    
    Logs:
    {tests_logs}
    """,
)

# Output the Gemini response
print(response.text)