import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import *

strategy = "SFStrategy"

# Recover all files in the directory io/back_test/
files = os.listdir(f"../io/back_test/strategies_time_evolution/{strategy}/")
# Dataframe to store the results
results = pd.DataFrame()

for file in files:
    results[file] = pd.read_csv(f"../io/back_test/strategies_time_evolution/{strategy}/" + file)["total_value"]

results["mean"] = results.mean(axis=1)
results["std"] = results.std(axis=1)

# Create a figure with a specific size
plt.figure(figsize=(8, 5))
plt.plot(results["mean"], label="Mean")
plt.fill_between(results.index, results["mean"] - results["std"], results["mean"] + results["std"], alpha=0.2)
plt.title("Back test results")
plt.xlabel("Time (minutes)")
plt.ylabel("Total value (USD)")
plt.legend()
plt.grid()
# Save the figure
plt.savefig(f"../io/back_test/fig/total_value_mean_std_{strategy}.pdf")

# plot the mean and the std
plt.figure(figsize=(8, 5))
# Plot all columns but the mean and the std, with reduced sickness
for col in results.columns:
    if col not in ["mean", "std"]:
        plt.plot(results[col], linewidth=0.35)
plt.title("Back test results")
plt.xlabel("Time (minutes)")
plt.ylabel("Total value (USD)")
# add grid
plt.grid()
# Save the figure
plt.savefig(f"../io/back_test/fig/total_value_all_{strategy}.pdf")

