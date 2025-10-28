import pandas as pd
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

# Create empty dataframe with 6 data points
df = pd.DataFrame(
    {
        "vina": [-9.725, -12.37, -11.398, -10.233, -7.1, -6.855],
        "boltz2": [-9.44, -7.24, -9.25, -7.85, -7.74, -8.59],
    }
)

# Calculate Spearman correlation
correlation, p_value = spearmanr(df["vina"], df["boltz2"])

# Create plot
plt.figure(figsize=(8, 6))
plt.scatter(df["vina"], df["boltz2"], alpha=0.6, s=100)
plt.xlabel("AutoDock Vina (kcal/mol)")
plt.ylabel("Boltz2 (kcal/mol)")
plt.title(f"AutoDock Vina vs Boltz2\nSpearman ρ = {correlation:.3f}")
plt.grid(True, alpha=0.3)

# Save figure
output_path = "./pipeline_output/docking/vina_vs_boltz2.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to {output_path}")
print(f"Spearman correlation: {correlation:.3f} (p-value: {p_value:.3e})")
