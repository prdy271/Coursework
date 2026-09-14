"""
Standalone Gemini connectivity test -- completely independent of the rest
of the app. Run this directly to check whether the problem is your API
key/model/network, or something in the app's own code.

Usage:
    python test_gemini.py

Run this from inside your activated venv (same one the app uses), in the
project folder, so it picks up the same .env file.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("FAIL: GEMINI_API_KEY not found in your .env file.")
    sys.exit(1)

masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "(too short?)"
print(f"Loaded API key: {masked}")

try:
    from google import genai
except ImportError:
    print("FAIL: google-genai package not installed. Run: pip install google-genai")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# ---------- Step 1: can we reach Gemini at all? (lists models -- tests auth/network, not a specific model) ----------
print("\n--- Step 1: Listing available models (tests API key + network, not any specific model) ---")
try:
    models = list(client.models.list())
    print(f"OK: connected successfully, {len(models)} models visible to this API key.")
except Exception as e:
    print(f"FAIL at Step 1: {type(e).__name__}: {e}")
    print("\nThis means the problem is your API key, network, or Google's API being down entirely --")
    print("not a specific model. Double check GEMINI_API_KEY in .env is correct and not expired.")
    sys.exit(1)

# ---------- Step 2: try each candidate model with a trivial prompt ----------
MODELS_TO_TEST = ["gemini-3.6-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]

for model_name in MODELS_TO_TEST:
    print(f"\n--- Step 2: Testing model '{model_name}' with a trivial prompt ---")
    for attempt in range(1, 3):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=model_name,
                contents="Reply with exactly one word: hello",
            )
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s): model replied: {response.text.strip()!r}")
            break
        except Exception as e:
            print(f"Attempt {attempt} FAILED: {type(e).__name__}: {e}")
            if attempt < 2:
                print("Retrying once more in 3 seconds...")
                time.sleep(3)
    else:
        print(f"'{model_name}' failed both attempts.")

print("\n--- Done ---")
print("If Step 1 succeeded but every model in Step 2 failed with the same 503/UNAVAILABLE")
print("error, this is genuinely Google's servers, not your code or setup.")
print("If Step 1 failed, it's your API key/network, not the model choice at all.")
print("If one model works but another doesn't, that specific model is the problem.")
