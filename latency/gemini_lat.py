import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent plt.show() errors
import matplotlib.pyplot as plt
from scipy.stats import norm
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import time
import base64
import os
import statistics
import httpx
import json
from typing import List, Dict, Any, Tuple, Optional

# Global flag to check if plotting libraries are available
_plotting_available = True
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend
    import matplotlib.pyplot as plt
    from scipy.stats import norm
except ImportError:
    print("Warning: Plotting libraries (numpy, matplotlib, scipy) not found.")
    print("Plotting will be disabled. Install with: pip install numpy matplotlib scipy")
    _plotting_available = False

# Global flag for Google vision
_google_vision_available = True
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL (Pillow) library not found.")
    print("Multimodal requests for Google Direct client will be disabled.")
    print("Install with: pip install Pillow")
    _google_vision_available = False


class TestConfig:
    """Configuration class for the latency test."""
    
    # Model and Prompt
    # Name for the proxy (e.g., what LiteLLM calls it)
    PROXY_MODEL = "gemini/gemini-2.5-flash" 
    # Official name for the Google API
    GOOGLE_MODEL = "gemini-2.5-flash" 
    PROMPT = "Hello, how are you today?"
    
    # Optional: Path to an image for multimodal requests
    # IMAGE_PATH = "/path/to/your/image.jpg"
    IMAGE_PATH = None

    # Test parameters
    NUM_REQUESTS_PER_EXP = 10  # Number of requests in each experiment (for avg)
    NUM_EXPERIMENTS = 5        # Number of experiments to run (for distribution)
    
    # Endpoints
    PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:4000")
    PROXY_KEY = os.environ.get("PROXY_API_KEY", "") # Key for your proxy
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # Reads from environment

    # Test Type
    STREAM = False # Set to True to test streaming (Time To First Token)
    
    # Other settings
    TIMEOUT = 120.0 # HTTP client timeout in seconds (for proxy client)
    SHOW_PLOT = True # Set to False to disable the final plot
    JSON_OUTPUT_FILE = "latency_results_gemini_2.json" # File to save all results
    PLOT_OUTPUT_FILE = "latency_plot_gemini_2.png" # File to save the plot


def encode_image(image_path: str) -> str:
    """
    Encodes an image file as a base64 string.
    
    Args:
        image_path: The path to the image file.

    Returns:
        A base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return "" # Return empty string to allow execution to continue
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

def prepare_proxy_messages(prompt: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Prepares the 'messages' payload for the OpenAI-compatible proxy.
    """
    if not image_path:
        return [{"role": "user", "content": prompt}]
    
    print("Image provided. Preparing multimodal request for Proxy (OpenAI format)...")
    base64_image = encode_image(image_path)
    if not base64_image:
        print("Warning: Could not encode image. Proceeding with text-only prompt.")
        return [{"role": "user", "content": prompt}]
        
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]

def prepare_google_payload(prompt: str, image_path: Optional[str] = None) -> List[Any]:
    """
    Prepares the payload for the google-generativeai client.
    """
    if not image_path:
        return [prompt] # Text-only payload
    
    if not _google_vision_available:
        print("Warning: PIL not found. Proceeding with text-only prompt for Google Direct.")
        return [prompt]
        
    print("Image provided. Preparing multimodal request for Google Direct...")
    try:
        img = Image.open(image_path)
        return [prompt, img] # Multimodal payload
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Proceeding with text-only.")
        return [prompt]
    except Exception as e:
        print(f"Error loading image with PIL: {e}. Proceeding with text-only.")
        return [prompt]

# --- Request Functions (OpenAI Proxy) ---

def make_proxy_request(client: openai.OpenAI, model: str, messages: List[Dict[str, Any]]) -> float:
    """Makes a single chat completion request to the proxy and returns latency."""
    try:
        start_time = time.perf_counter()
        client.chat.completions.create(model=model, messages=messages)
        end_time = time.perf_counter()
        return end_time - start_time
    except openai.APIError as e:
        print(f"  -> Proxy request failed: {e.__class__.__name__} - {e.message}")
        return float('inf')
    except Exception as e:
        print(f"  -> An unexpected error occurred with proxy: {e}")
        return float('inf')

def make_proxy_streaming_request(client: openai.OpenAI, model: str, messages: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Makes a streaming proxy request and measures TTFT and total latency."""
    try:
        start_time = time.perf_counter()
        first_token_time = None
        
        response = client.chat.completions.create(model=model, messages=messages, stream=True)
        
        for chunk in response:
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = time.perf_counter()
        
        end_time = time.perf_counter()

        ttft = (first_token_time - start_time) if first_token_time else float('inf')
        total_latency = end_time - start_time
        return ttft, total_latency
    except openai.APIError as e:
        print(f"  -> Proxy streaming request failed: {e.__class__.__name__} - {e.message}")
        return float('inf'), float('inf')
    except Exception as e:
        print(f"  -> An unexpected error occurred during proxy streaming: {e}")
        return float('inf'), float('inf')

# --- Request Functions (Google Direct) ---

def make_google_request(model: genai.GenerativeModel, payload: List[Any]) -> float:
    """Makes a single request to Google and returns latency."""
    try:
        start_time = time.perf_counter()
        # Use GenerationConfig to prevent function calling, safety settings, etc.
        model.generate_content(payload, generation_config=GenerationConfig(candidate_count=1))
        end_time = time.perf_counter()
        return end_time - start_time
    except Exception as e:
        print(f"  -> Google Direct request failed: {e}")
        return float('inf')

def make_google_streaming_request(model: genai.GenerativeModel, payload: List[Any]) -> Tuple[float, float]:
    """Makes a streaming Google request and measures TTFT and total latency."""
    try:
        start_time = time.perf_counter()
        first_token_time = None
        
        response = model.generate_content(payload, stream=True, generation_config=GenerationConfig(candidate_count=1))
        
        for chunk in response:
            if first_token_time is None and chunk.text:
                first_token_time = time.perf_counter()
        
        end_time = time.perf_counter()

        ttft = (first_token_time - start_time) if first_token_time else float('inf')
        total_latency = end_time - start_time
        return ttft, total_latency
    except Exception as e:
        print(f"  -> An unexpected error occurred during Google streaming: {e}")
        return float('inf'), float('inf')

# --- Test Execution ---

def run_proxy_tests(
    client: openai.OpenAI,
    client_name: str,
    model: str,
    messages: List[Dict[str, Any]],
    num_requests: int,
    stream: bool
) -> Tuple[Dict[str, float], List[float]]:
    """Runs latency tests for the OpenAI-compatible proxy."""
    latencies = []
    
    for i in range(num_requests):
        if stream:
            ttft, _ = make_proxy_streaming_request(client, model, messages)
            latency = ttft
        else:
            latency = make_proxy_request(client, model, messages)
        
        if latency != float('inf'):
            latencies.append(latency)

    if not latencies:
        print(f"  All requests failed for {client_name}. Cannot calculate statistics.")
        return {}, []

    stats = {
        "avg": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "successful_requests": len(latencies),
    }
    
    return stats, latencies

def run_google_tests(
    model: genai.GenerativeModel,
    client_name: str,
    payload: List[Any],
    num_requests: int,
    stream: bool
) -> Tuple[Dict[str, float], List[float]]:
    """Runs latency tests for the Google Direct client."""
    latencies = []
    
    for i in range(num_requests):
        if stream:
            ttft, _ = make_google_streaming_request(model, payload)
            latency = ttft
        else:
            latency = make_google_request(model, payload)
        
        if latency != float('inf'):
            latencies.append(latency)

    if not latencies:
        print(f"  All requests failed for {client_name}. Cannot calculate statistics.")
        return {}, []

    stats = {
        "avg": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "successful_requests": len(latencies),
    }
    
    return stats, latencies

# --- Aggregation, Plotting, and Saving ---

def aggregate_stats(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregates statistics from a list of test runs."""
    if not results_list:
        return {}
    
    # Use 'avg' (mean latency from each run) for calculating overall distribution
    all_means = [r['avg'] for r in results_list if 'avg' in r]
    
    return {
        "avg": statistics.mean(all_means) if all_means else 0,
        "median": statistics.median(all_means) if all_means else 0,
        "min": min([r['min'] for r in results_list if 'min' in r]),
        "max": max([r['max'] for r in results_list if 'max' in r]),
        "stdev": statistics.stdev(all_means) if len(all_means) > 1 else 0,
        "successful_requests": sum([r['successful_requests'] for r in results_list if 'successful_requests' in r])
    }

def plot_distributions(proxy_means: List[float], google_means: List[float], model_name: str, stream: bool, num_experiments: int, config: TestConfig):
    """Plots the normal distribution of the mean latencies."""
    if not _plotting_available:
        print("\nPlotting skipped as required libraries are not installed.")
        return
    
    if not proxy_means or not google_means:
        print("\nPlotting skipped due to lack of successful experiment data.")
        return
        
    # Calculate parameters for the normal distributions
    proxy_mu = statistics.mean(proxy_means)
    proxy_sigma = statistics.stdev(proxy_means) if len(proxy_means) > 1 else 1e-9
    
    google_mu = statistics.mean(google_means)
    google_sigma = statistics.stdev(google_means) if len(google_means) > 1 else 1e-9

    # Set up plot range
    min_val = min(min(proxy_means), min(google_means)) - 3 * max(proxy_sigma, google_sigma)
    max_val = max(max(proxy_means), max(google_means)) + 3 * max(proxy_sigma, google_sigma)
    x = np.linspace(min_val, max_val, 300)

    # Calculate Probability Density Functions (PDF)
    proxy_pdf = norm.pdf(x, proxy_mu, proxy_sigma)
    google_pdf = norm.pdf(x, google_mu, google_sigma)

    # Plot
    fig = plt.figure(figsize=(12, 7))
    
    # Plot histograms of the actual collected means
    plt.hist(proxy_means, bins=20, density=True, alpha=0.6, color='blue', label="Proxy Means (Actual Data)")
    plt.hist(google_means, bins=20, density=True, alpha=0.6, color='red', label="Google Direct Means (Actual Data)")
    
    # Plot the fitted normal distributions
    plt.plot(x, proxy_pdf, 'b-', linewidth=2, label=f"Proxy Fit: $\\mu={proxy_mu:.3f}s, \\sigma={proxy_sigma:.3f}s$")
    plt.plot(x, google_pdf, 'r-', linewidth=2, label=f"Google Direct Fit: $\\mu={google_mu:.3f}s, \\sigma={google_sigma:.3f}s$")

    metric_name = "TTFT (Time To First Token)" if stream else "Total Response Time"
    plt.title(f"Distribution of Mean Latencies ({num_experiments} Experiments)\nModel: {model_name}")
    plt.xlabel(f"Mean {metric_name} (seconds)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot to a file
    try:
        plt.savefig(config.PLOT_OUTPUT_FILE)
        print(f"\nSuccessfully saved plot to {config.PLOT_OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    
    plt.close(fig) # Close the figure to free memory


def print_results(results: Dict[str, Dict[str, Any]]):
    """Prints a formatted comparison of the test results."""
    print("\n" + "="*40)
    print(" " * 12 + "LATENCY RESULTS")
    print("="*40)

    metric_name = "TTFT (Time To First Token)" if "Stream" in results else "Total Response Time"
    
    headers = ["Metric", "Proxy", "Google Direct", "Difference"]
    print(f"{headers[0]:<12} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12}")
    print("-" * 55)

    proxy_results = results.get("Proxy", {}) or results.get("Proxy Stream", {})
    google_results = results.get("Google Direct", {}) or results.get("Google Direct Stream", {})
    
    for key in ["avg", "median", "min", "max", "stdev"]:
        proxy_val = proxy_results.get(key, float('nan'))
        google_val = google_results.get(key, float('nan'))
        diff = proxy_val - google_val
        print(f"{key.capitalize():<12} | {proxy_val:<12.3f}s | {google_val:<12.3f}s | {diff:<+12.3f}s")
    
    print("-" * 55)
    proxy_success = proxy_results.get('successful_requests', 0)
    google_success = google_results.get('successful_requests', 0)
    print(f"{'Successful':<12} | {proxy_success:<12} | {google_success:<12} |")
    print("="*40)

def save_results_to_json(
    config: TestConfig,
    final_stats: Dict[str, Any],
    proxy_exp_stats: List[Dict[str, float]],
    google_exp_stats: List[Dict[str, float]],
    proxy_raw_data: List[List[float]],
    google_raw_data: List[List[float]]
):
    """Saves all configuration, summary, and raw data to a JSON file."""
    
    data_to_save = {
        "test_configuration": {
            "proxy_model": config.PROXY_MODEL,
            "google_model": config.GOOGLE_MODEL,
            "prompt_length": len(config.PROMPT),
            "image_provided": config.IMAGE_PATH is not None,
            "num_requests_per_exp": config.NUM_REQUESTS_PER_EXP,
            "num_experiments": config.NUM_EXPERIMENTS,
            "stream": config.STREAM,
            "timeout": config.TIMEOUT,
            "proxy_url": config.PROXY_URL,
        },
        "aggregated_summary": final_stats,
        "proxy_results": {
            "experiment_stats": proxy_exp_stats,
            "raw_latencies_per_experiment": proxy_raw_data
        },
        "google_results": {
            "experiment_stats": google_exp_stats,
            "raw_latencies_per_experiment": google_raw_data
        }
    }
    
    try:
        with open(config.JSON_OUTPUT_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"\nSuccessfully saved detailed results to {config.JSON_OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError saving results to JSON: {e}")

# --- Main Execution ---

def run_latency_comparison(config: TestConfig):
    """Main function to run the latency comparison based on the config."""

    if not config.GOOGLE_API_KEY:
        print("Error: Google API key is required. Set the GOOGLE_API_KEY environment variable.")
        return

    # Prepare payloads
    proxy_messages = prepare_proxy_messages(config.PROMPT, config.IMAGE_PATH)
    google_payload = prepare_google_payload(config.PROMPT, config.IMAGE_PATH)
    
    timeout = httpx.Timeout(config.TIMEOUT)

    # Setup clients
    print(f"Setting up Proxy client for: {config.PROXY_URL}")
    proxy_client = openai.OpenAI(api_key=config.PROXY_KEY, base_url=config.PROXY_URL, timeout=timeout)
    
    print("Setting up Google Direct client...")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    google_client = genai.GenerativeModel(config.GOOGLE_MODEL)

    # --- Run multiple experiments ---
    all_proxy_results = []
    all_google_results = []
    all_proxy_raw_latencies = []
    all_google_raw_latencies = []
    
    client_name_suffix = " Stream" if config.STREAM else ""
    
    print(f"Starting {config.NUM_EXPERIMENTS} experiments, each with {config.NUM_REQUESTS_PER_EXP} requests...")
    
    for i in range(config.NUM_EXPERIMENTS):
        print(f"\n--- Experiment {i + 1}/{config.NUM_EXPERIMENTS} ---")
        
        print(f"Testing Proxy ({config.PROXY_MODEL})...")
        proxy_stats, proxy_raw = run_proxy_tests(
            proxy_client, f"Proxy{client_name_suffix}", 
            config.PROXY_MODEL, proxy_messages, config.NUM_REQUESTS_PER_EXP, config.STREAM
        )
        if proxy_stats:
            all_proxy_results.append(proxy_stats)
            all_proxy_raw_latencies.append(proxy_raw)
            
        print(f"Testing Google Direct ({config.GOOGLE_MODEL})...")
        google_stats, google_raw = run_google_tests(
            google_client, f"Google Direct{client_name_suffix}",
            google_payload, config.NUM_REQUESTS_PER_EXP, config.STREAM
        )
        if google_stats:
            all_google_results.append(google_stats)
            all_google_raw_latencies.append(google_raw)

    if not all_proxy_results or not all_google_results:
        print("\nNo successful experiments were completed. Exiting.")
        return

    # --- Aggregate and Print Final Table ---
    final_proxy_stats = aggregate_stats(all_proxy_results)
    final_google_stats = aggregate_stats(all_google_results)
    
    results_for_table = {
        f"Proxy{client_name_suffix}": final_proxy_stats,
        f"Google Direct{client_name_suffix}": final_google_stats
    }
    
    print_results(results_for_table)

    # --- Plot Distributions ---
    if config.SHOW_PLOT:
        proxy_means = [r['avg'] for r in all_proxy_results if 'avg' in r]
        google_means = [r['avg'] for r in all_google_results if 'avg' in r]
        plot_distributions(proxy_means, google_means, config.PROXY_MODEL, config.STREAM, config.NUM_EXPERIMENTS, config)
        
    # --- Save Detailed JSON Output ---
    save_results_to_json(
        config,
        results_for_table,
        all_proxy_results,
        all_google_results,
        all_proxy_raw_latencies,
        all_google_raw_latencies
    )

def main_run():
    
    print("Starting Latency Comparison Test (Proxy Gemini vs. Direct Gemini)...")
    
    # 1. Create a config instance
    config = TestConfig()
    
    # 2. (Optional) Override config if needed, e.g.:
    # config.STREAM = True
    # config.NUM_EXPERIMENTS = 20
    
    # 3. Run the test
    run_latency_comparison(config)

if __name__ == "__main__":
    main_run()