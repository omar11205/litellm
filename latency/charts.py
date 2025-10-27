import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os  # Import os to check if file exists
import pandas
import matplotlib
import seaborn

# --- create_latency_visualizations ---
# This function now loads from a file.
def create_latency_visualizations(
    json_filename: str = "latency_results_20251027_080128.json",
    output_filename: str = "latency_comparison.png"
):
    """
    Loads latency data from a JSON file and generates a box plot and
    a distribution plot, saving them to a file.

    Requires: pandas, matplotlib, seaborn
    """

    print(f"Loading and parsing JSON data from '{json_filename}'...")
    try:
        # Check if file exists
        if not os.path.exists(json_filename):
            print(f"Error: The file '{json_filename}' was not found.")
            print(f"Please make sure it's in the same directory as this script.")
            return

        # Open and load the JSON file
        with open(json_filename, 'r') as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data in '{json_filename}'. {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # --- 1. Prepare Data for Seaborn ---
    # Convert the dictionary of lists into a "long-form" DataFrame.
    # This is the format Seaborn (and most statistical plotting) prefers.
    plot_data = []
    for group_name, latencies in data.items():
        # Clean up group names for better plotting
        # e.g., "Group 1: Direct OpenAI (gpt-4o-mini)" -> "Group 1:\nDirect OpenAI..."
        clean_name = group_name.replace(": ", ":\n")
        for latency in latencies:
            plot_data.append({
                "Group": clean_name,
                "Latency (ms)": latency
            })

    if not plot_data:
        print("Error: No data to plot.")
        return

    df = pd.DataFrame(plot_data)

    # --- 2. Create Visualizations ---
    print("Generating visualizations...")

    # Set the style
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Create a figure with two subplots, side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 9)) # Increased height for taller y-axis labels

    # --- Plot 1: Box Plot (for statistical summary) ---
    # A box plot is the best way to compare distributions and see outliers.
    
    # FIX 1: Added hue="Group" and legend=False to address the FutureWarning
    sns.boxplot(
        data=df,
        x="Latency (ms)",
        y="Group",
        hue="Group",
        ax=axes[0],
        orient="h",
        palette="Blues",
        legend=False
    )
    axes[0].set_title("Latency Comparison (Box Plot)", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Latency (ms)", fontsize=12)
    axes[0].set_ylabel("", fontsize=12)  # Remove y-label, group names are clear
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    
    # FIX 2: Changed 'linestyle=False' to 'visible=False' to fix the ValueError
    axes[0].grid(axis='y', visible=False)  # Turn off horizontal grid lines
    
    axes[0].tick_params(axis='y', labelsize=11) # Adjust y-tick label size

    # --- Plot 2: Distribution Plot (Histogram/KDE) ---
    # This shows the *shape* of the latency distribution (e.g., is it spiky?)
    sns.histplot(
        data=df,
        x="Latency (ms)",
        hue="Group",
        kde=True,  # Kernel Density Estimate (smooth line)
        multiple="layer",  # Overlay the distributions
        ax=axes[1],
        palette="muted",
        bins=50,
        legend=True
    )
    axes[1].set_title("Latency Distribution (Histogram + KDE)", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Latency (ms)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    
    # Improve legend
    legend = axes[1].get_legend()
    if legend:
        legend.set_title("Group")
        plt.setp(legend.get_title(), fontsize='13')
        plt.setp(legend.get_texts(), fontsize='11')


    # --- Final Touches ---
    fig.suptitle("API Latency Experiment Analysis", fontsize=22, fontweight='bold', y=1.03)
    plt.tight_layout()  # Adjusts plots to prevent overlap

    # --- 3. Save and Show ---
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\n[SUCCESS] Visualization saved to '{output_filename}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save the plot: {e}")

def main():
    # Check for required libraries
    try:
        import pandas
        import matplotlib
        import seaborn
    except ImportError as e:
        print(f"Error: Missing required library: '{e.name}'")
        print(f"Please install it by running: pip install {e.name}")
        return

    # Now calls the function to load from the default file 'latency_results.json'
    create_latency_visualizations()

if __name__ == "__main__":
    main()