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

# Recover all files in the directory io/back_test/
files = os.listdir("../io/back_test/strategies_time_evolution/")
# Dataframe to store the results
results = pd.DataFrame()

for file in files:
    results[file] = pd.read_csv("../io/back_test/strategies_time_evolution/" + file)["total_value"]

results["mean"] = results.mean(axis=1)
results["std"] = results.std(axis=1)
# plot the mean and the std
results["mean"].plot()
plt.fill_between(results.index, results["mean"] - results["std"], results["mean"] + results["std"], alpha=0.2)
plt.title("Mean of the back test results")
plt.xlabel("Time")
plt.ylabel("Total value")
# add grid
plt.grid()
plt.show()

# Plot the results but the
results.plot()
# Remove legend
plt.legend().remove()
plt.title("Back test results")
plt.xlabel("Time")
plt.ylabel("Total value")
plt.show()