import openai
import google.generativeai as genai
import os
import time

# --- Configuration ---
# 1. Set your proxy URL (e.g., LiteLLM)
PROXY_URL = "http://localhost:4000"

# 2. Set the API key for your proxy.
PROXY_API_KEY = ""

# 3. Specify the Gemini model you are proxying (as named by your proxy)
#    Note: "gemini/gemini-2.5-flash" is from your original script.
PROXY_MODEL_NAME = os.getenv("PROXY_MODEL_NAME", "gemini/gemini-2.5-flash")

# 4. Set your official Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# 5. Specify the official Google model name
#    Note: The current Flash model is "gemini-1.5-flash".
DIRECT_MODEL_NAME = os.getenv("DIRECT_MODEL_NAME", "gemini-2.5-flash")
# --- End Configuration ---


# ==========================================
# PROXY CLIENT (via openai library)
# ==========================================

def create_proxy_client(base_url: str, api_key: str) -> openai.OpenAI:
    """
    Creates an OpenAI client configured to talk to a proxy.
    """
    print(f"Initializing proxy client for: {base_url}")
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
    )

def query_gemini_via_proxy(client: openai.OpenAI, model: str, prompt: str):
    """
    Sends a request to the proxy server using the OpenAI-compatible format.
    """
    print(f"\nSending request to model: {model}")
    print(f"User Prompt: {prompt}")
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        start_time = time.perf_counter()
        
        # This is the actual API call to your proxy
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        content = response.choices[0].message.content
        
        print("\n--- Response Received ---")
        print(f"Latency: {latency:.3f} seconds")
        print(f"Response: {content.strip()}")
        
    except openai.APIConnectionError as e:
        print(f"\n--- ERROR ---")
        print(f"Failed to connect to the proxy at: {PROXY_URL}")
        print("Please ensure your proxy server (e.g., LiteLLM) is running.")
        print(f"Details: {e}")
    except openai.APIError as e:
        print(f"\n--- API ERROR ---")
        print(f"The proxy reported an error: {e}")
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---")
        print(f"An error occurred: {e}")

# ==========================================
# DIRECT CLIENT (via google-generativeai)
# ==========================================

def create_google_client(api_key: str) -> genai.GenerativeModel:
    """
    Configures the Google Generative AI client and returns a model instance.
    """
    print("Initializing Google direct client...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(DIRECT_MODEL_NAME)
    return model

def query_gemini_direct(model: genai.GenerativeModel, prompt: str):
    """
    Sends a request directly to the Google API.
    """
    print(f"\nSending request to model: {model.model_name}")
    print(f"User Prompt: {prompt}")
    
    try:
        start_time = time.perf_counter()
        
        # This is the actual API call to Google
        response = model.generate_content(prompt)
        
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        content = response.text
        
        print("\n--- Response Received ---")
        print(f"Latency: {latency:.3f} seconds")
        print(f"Response: {content.strip()}")

    except Exception as e:
        print(f"\n--- GOOGLE API ERROR ---")
        print(f"An error occurred while contacting the Google API: {e}")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    test_prompt = "Hello! Tell me a one-sentence fact about the planet Mars."
    
    # --- Test 1: Proxy ---
    print("\n--- START: PROXY TEST (via openai library) ---")
    if not PROXY_URL or not PROXY_API_KEY:
        print("PROXY_URL or PROXY_API_KEY environment variables not set. Skipping proxy test.")
    else:
        proxy_client = create_proxy_client(base_url=PROXY_URL, api_key=PROXY_API_KEY)
        query_gemini_via_proxy(proxy_client, PROXY_MODEL_NAME, test_prompt)
    print("\n--- END: PROXY TEST ---")
    
    
    # --- Test 2: Direct API ---
    print("\n\n--- START: DIRECT API TEST (via google-generativeai library) ---")
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY environment variable not set. Skipping direct API test.")
        print("Get a key from: https://aistudio.google.com/app/apikey")
    else:
        direct_model = create_google_client(api_key=GOOGLE_API_KEY)
        query_gemini_direct(direct_model, test_prompt)
    print("\n--- END: DIRECT API TEST ---")


if __name__ == "__main__":
    main()
