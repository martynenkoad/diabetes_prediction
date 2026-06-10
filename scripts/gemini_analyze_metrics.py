import json

from google import genai
import os

# Access the Gemini API key
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Load analyze metrics prompt
with open("prompts/analyze-metrics.txt") as f:
    prompt = f.read()

# Load evaluation metrics
with open("evaluation_results.json") as f:
    metrics = json.load(f)

metrics_text = json.dumps(
    metrics,
    indent=2
)

# Load metrics logs
with open("metrics_logs.txt") as f:
    metrics_logs = f.read()

# Send the request to the Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""
    {prompt}
    
    Metrics:
    {metrics_text}
    
    Logs:
    {metrics_logs}
    """,
)

# Output the Gemini response
print(response.text)