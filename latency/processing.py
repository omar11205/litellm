from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Process:
    data_list: List[str]
    all_data: List[Dict[str, Any]] = field(default_factory=list)

    def load(self) -> None:
        """Load all JSON files specified in data_list."""
        for data_route in self.data_list:
            with open(data_route, "r") as f:
                data = json.load(f)
                self.all_data.append(data)

    def unify_latencies(self, input_file: str, output_file: str) -> None:
        """Load a single JSON, unify latency lists, and save results."""
        with open(input_file, "r") as f:
            data = json.load(f)

        proxy_latencies = data["proxy_results"]["raw_latencies_per_experiment"]
        openai_latencies = data["google_results"]["raw_latencies_per_experiment"]

        data["total_openai_latencies"] = [
            lat for exp in openai_latencies for lat in exp
        ]
        data["total_proxy_latencies"] = [
            lat for exp in proxy_latencies for lat in exp
        ]

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Unified latency results saved to {output_file}.")

    def join_total(self, field: str) -> List[Any]:
        """Join a specific field from all loaded data files."""
        return [data[field] for data in self.all_data]


@dataclass
class Analysis:
    """
    Performs statistical analysis and visualization on groups of data.
    
    Expects data_groups to be a list of lists, e.g., [[group1_data], [group2_data]]
    
    Requires 'numpy', 'scipy', 'pandas', 'matplotlib', and 'seaborn' to be installed:
    pip install numpy scipy pandas matplotlib seaborn
    """
    data_groups: List[List[float]]
    results: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self, group1_idx: int = 0, group2_idx: int = 1) -> None:
        """Prints a descriptive statistics summary for two groups."""
        if not self._check_indices(group1_idx, group2_idx):
            return

        group1 = self.data_groups[group1_idx]
        group2 = self.data_groups[group2_idx]
        
        print("--- Statistical Summary ---")
        print(f"\n LiteLLM Proxy gpt-4o-mini Latencies:")
        print(f"  Count:  {len(group1)}")
        print(f"  Mean:   {np.mean(group1):.4f}")
        print(f"  Median: {np.median(group1):.4f}")
        print(f"  StdDev: {np.std(group1):.4f}")
        print(f"  Min:    {np.min(group1):.4f}")
        print(f"  Max:    {np.max(group1):.4f}")
        
        print(f"\nOpenAI Python API gpt-4o-mini Latencies:")
        print(f"  Count:  {len(group2)}")
        print(f"  Mean:   {np.mean(group2):.4f}")
        print(f"  Median: {np.median(group2):.4f}")
        print(f"  StdDev: {np.std(group2):.4f}")
        print(f"  Min:    {np.min(group2):.4f}")
        print(f"  Max:    {np.max(group2):.4f}")
        print("---------------------------\n")

    def compare_distributions(self, group1_idx: int = 0, group2_idx: int = 1, alpha: float = 0.05) -> None:
        """
        Performs a Mann-Whitney U test to compare two independent distributions.
        """
        if not self._check_indices(group1_idx, group2_idx):
            return

        group1 = self.data_groups[group1_idx]
        group2 = self.data_groups[group2_idx]

        if not group1 or not group2:
            print("Error: One or both data groups are empty. Cannot perform test.")
            return

        # Perform the Mann-Whitney U test
        # 'alternative='two-sided'' checks if the distributions are simply different.
        # Use 'less' to test if group1 is stochastically less than group2.
        # Use 'greater' to test if group1 is stochastically greater than group2.
        try:
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            self.results = {
                'test_type': 'Mann-Whitney U',
                'group1_index': group1_idx,
                'group2_index': group2_idx,
                'statistic': stat,
                'p_value': p_value,
                'alpha': alpha
            }

            print("--- Mann-Whitney U Test Results ---")
            print(f"Comparing Group {group1_idx} and Group {group2_idx}")
            print(f"Statistic: {stat:.4f}")
            print(f"P-Value:   {p_value:.6f}")

            if p_value < alpha:
                print(f"\nConclusion: The difference is statistically significant (p < {alpha}).")
                print("We reject the null hypothesis; the two distributions are different.")
            else:
                print(f"\nConclusion: The difference is NOT statistically significant (p >= {alpha}).")
                print("We fail to reject the null hypothesis; we cannot conclude the distributions are different.")
            print("-----------------------------------\n")

        except ValueError as e:
            print(f"Error during statistical test: {e}")
        except ImportError:
            print("Error: 'scipy' library not found. Please run 'pip install scipy'")

    def plot_distributions(
        self, 
        group1_idx: int = 0, 
        group2_idx: int = 1, 
        labels: List[str] = None, 
        save_path: str = None
    ) -> None:
        """
        Generates a violin plot to visually compare two distributions.
        Can optionally save the plot to a file.
        """
        if not self._check_indices(group1_idx, group2_idx):
            return
            
        try:
            # Set default labels if not provided
            if labels is None or len(labels) < 2:
                label1, label2 = f"Group {group1_idx}", f"Group {group2_idx}"
            else:
                label1, label2 = labels[0], labels[1]

            # Prepare data for Seaborn (long-form DataFrame)
            group1_data = self.data_groups[group1_idx]
            group2_data = self.data_groups[group2_idx]

            data_to_plot = []
            for val in group1_data:
                data_to_plot.append({'Latency': val, 'Group': label1})
            for val in group2_data:
                data_to_plot.append({'Latency': val, 'Group': label2})
            
            df = pd.DataFrame(data_to_plot)

            # Create the plot
            plt.figure(figsize=(10, 7))
            sns.set_theme(style="whitegrid")
            
            # Draw the violin plot
            sns.violinplot(
                x='Group', 
                y='Latency', 
                data=df, 
                inner='quartile',  # Shows median and quartiles
                cut=0,             # Doesn't extend past data
                palette="muted"
            )
            
            # Add individual data points with jitter
            sns.stripplot(
                x='Group', 
                y='Latency', 
                data=df, 
                color='black', 
                alpha=0.3, 
                jitter=True,
                size=3
            )

            plt.title('Latency Distribution Comparison', fontsize=16)
            plt.ylabel('Latency (s)', fontsize=12)
            plt.xlabel('Experiment Group', fontsize=12)
            
            # Save the plot if a path is provided
            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {save_path}")
                except Exception as e:
                    print(f"Error saving plot to {save_path}: {e}")

            # Show the plot in a new window
            print("Displaying plot... (Close the plot window to continue script)")
            plt.show()

        except ImportError:
            print("Error: 'pandas', 'matplotlib', or 'seaborn' not found.")
            print("Please run: pip install pandas matplotlib seaborn")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")


    def _check_indices(self, idx1: int, idx2: int) -> bool:
        """Helper to validate data group indices."""
        max_idx = len(self.data_groups) - 1
        if not (0 <= idx1 <= max_idx and 0 <= idx2 <= max_idx):
            print(f"Error: Invalid indices. Must be between 0 and {max_idx}.")
            return False
        if idx1 == idx2:
            print("Error: Cannot compare a group to itself. Indices must be different.")
            return False
        return True


if __name__ == "__main__":
    # Example usage:
    process = Process(data_list=["results_unified.json", "results_unified_1.json"])
    process.load()
    #process.unify_latencies("latency_results_gemini_2.json", "results_unified_gemini_2.json")
    total_proxy = process.join_total("total_proxy_latencies")
    total_openai = process.join_total("total_openai_latencies")

    combined = [total_proxy[0], total_openai[0]]

    if len(total_proxy) >= 2:
        analyzer_files = Analysis(data_groups=combined)
        analyzer_files.get_summary()
        analyzer_files.compare_distributions()
        analyzer_files.plot_distributions(
            labels=["LiteLLM Proxy gpt-4o-mini", "OpenAI Python API gpt-4o-mini"], 
            save_path="gpt-4o-mini_latency_comparison.png"
        )
    else:
        print("Could not find at least two 'total_proxy_latencies' lists to compare.")