import json
import matplotlib.pyplot as plt


# Load your JSON data (or use the dict directly if you already have it in memory)
with open("results_unified_1.json", "r") as f:
    data = json.load(f)

dataset_list = ["openai", "proxy"]

for dat in dataset_list:
    dataset = dat

    # Extract the unified latencies list
    latencies = data[f"total_{dataset}_latencies"]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Raw Latencies ({dataset})")

    # Optional: grid appearance
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Save to PNG
    output_file = f"latency_histogram_{dataset}_1.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to {output_file}")