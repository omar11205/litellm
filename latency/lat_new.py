import time
import os
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any
import json  # Import json for saving results
import datetime  # Import datetime for unique filenames

# --- Configuration ---
NUM_SAMPLES = 100  # Number of test requests per group. 100 is a good start.
NUM_WARMUP = 5     # Number of warmup requests to discard.

# PAYLOAD: This is the prompt that will be sent.
# We use a consistent prompt and token count to ensure a fair test.
TEST_PAYLOAD = {
    "messages": [{"role": "user", "content": "What are the top 5 most populated cities in the world?"}],
    "max_tokens": 150,
    "temperature": 0.5
}

# --- API Clients ---

# Client 1: Direct to OpenAI
# Make sure OPENAI_API_KEY is set in your environment
try:
    client_direct_openai = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
except Exception as e:
    print(f"Failed to initialize Direct OpenAI client: {e}")
    print("Please set your OPENAI_API_KEY environment variable.")
    exit()

# Client 2: Connects to LiteLLM Proxy
# API key can be anything for localhost, but is required by the library.
try:
    client_litellm_proxy = OpenAI(
        base_url="http://localhost:4000",
        api_key=""
    )
except Exception as e:
    print(f"Failed to initialize LiteLLM client: {e}")
    print("Is the LiteLLM Docker container running on port 4000?")
    exit()

# --- Test Definitions ---
# This list defines the 3 experimental groups.
# 'client' is the client object to use.
# 'model' is the model string to pass in the request.
# 'gpt-4o-mini-proxied' and 'gemini-2.5-flash' must match your LiteLLM config.yaml
EXPERIMENTAL_GROUPS = [
    {
        "name": "Group 1: Direct OpenAI (gpt-4o-mini)",
        "client": client_direct_openai,
        "model": "gpt-4o-mini"
    },
    {
        "name": "Group 2: Proxied OpenAI (gpt-4o-mini)",
        "client": client_litellm_proxy,
        "model": "gpt-4o-mini" # Must match config.yaml
    },
    {
        "name": "Group 3: Proxied Gemini (gemini-2.5-flash)",
        "client": client_litellm_proxy,
        "model": "gemini-2.5-flash" # Must match config.yaml
    }
]

def run_latency_test(
    client: OpenAI,
    model_name: str,
    payload: Dict[str, Any],
    num_samples: int
) -> List[float]:
    """
    Runs a latency test for a given client and model.

    Args:
        client: The OpenAI client instance (either direct or proxied).
        model_name: The name of the model to test.
        payload: The request body to send.
        num_samples: The number of samples to collect.

    Returns:
        A list of latency measurements in milliseconds.
    """
    latencies_ms = []
    for i in range(num_samples):
        try:
            start_time = time.perf_counter()

            # --- The API Call ---
            client.chat.completions.create(
                model=model_name,
                messages=payload["messages"],
                max_tokens=payload["max_tokens"],
                temperature=payload["temperature"]
            )
            # ---------------------

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies_ms.append(latency_ms)

            print(f"  Sample {i+1}/{num_samples}: {latency_ms:.2f} ms")
            
            # Add a small delay to avoid hitting rate limits too hard
            time.sleep(0.1) 

        except Exception as e:
            print(f"  Sample {i+1}/{num_samples}: FAILED. Error: {e}")
            
    return latencies_ms

def print_statistics(name: str, latencies: List[float]):
    """Calculates and prints percentile statistics for a list of latencies."""
    if not latencies:
        print(f"\n--- Statistics for {name} ---")
        print("  No successful requests to analyze.")
        print("-" * 40)
        return

    # Using numpy for easy percentile calculation
    lat_array = np.array(latencies)

    print(f"\n--- Statistics for {name} ---")
    print(f"  Successful Samples: {len(lat_array)}")
    print(f"  Mean (Average):     {np.mean(lat_array):.2f} ms")
    print(f"  Median (P50):       {np.median(lat_array):.2f} ms")
    print(f"  Std. Deviation:     {np.std(lat_array):.2f} ms")
    print(f"  Min:                {np.min(lat_array):.2f} ms")
    print(f"  Max:                {np.max(lat_array):.2f} ms")
    print(f"  P90 (90th %ile):    {np.percentile(lat_array, 90):.2f} ms")
    print(f"  P95 (95th %ile):    {np.percentile(lat_array, 95):.2f} ms")
    print(f"  P99 (99th %ile):    {np.percentile(lat_array, 99):.2f} ms")
    print("-" * 40)

def save_results_to_json(results: Dict[str, List[float]]):
    """Saves the complete results dictionary to a timestamped JSON file."""
    
    # Generate a unique filename with the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"latency_results_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n[SUCCESS] All latency data saved to {filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save results to JSON: {e}")

def main():
    print("Starting Latency Experiment...")
    print(f"Running {NUM_SAMPLES} samples per group with {NUM_WARMUP} warmup runs.")
    
    all_results = {}

    for group in EXPERIMENTAL_GROUPS:
        name = group["name"]
        client = group["client"]
        model = group["model"]
        
        print(f"\n--- Warming up {name} ---")
        run_latency_test(client, model, TEST_PAYLOAD, NUM_WARMUP)
        print("--- Warmup complete ---")

        print(f"\n--- Running Test: {name} ---")
        latencies = run_latency_test(client, model, TEST_PAYLOAD, NUM_SAMPLES)
        all_results[name] = latencies

    # --- Analysis Phase ---
    print("\n\n" + "=" * 40)
    print("      Latency Experiment Complete: Results")
    print("=" * 40)
    
    for name, results in all_results.items():
        print_statistics(name, results)
        
    # --- Save Results Phase ---
    save_results_to_json(all_results)

if __name__ == "__main__":
    # Check for required libraries
    try:
        import numpy
        import openai
    except ImportError as e:
        print(f"Error: Missing required library. Please install it: pip install {e.name}")
        exit()

    main()