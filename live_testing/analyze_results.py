import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)


# Load the data
data_1 = pd.read_csv('../io/live_test/perf/SFStrategy-17-05-2024-23-21-04SFStrategy__0.01.csv')
data_2 = pd.read_csv('../io/live_test/perf/SFStrategy-17-05-2024-23-21-10SFStrategy__0.005.csv')
data_3 = pd.read_csv('../io/live_test/perf/SFStrategy-17-05-2024-23-21-15SFStrategy__0.001.csv')
data_4 = pd.read_csv('../io/live_test/perf/SFStrategyI-17-05-2024-23-20-44SFStrategyI__0.01.csv')
data_5 = pd.read_csv('../io/live_test/perf/SFStrategyI-17-05-2024-23-20-49SFStrategyI__0.005.csv')
data_6 = pd.read_csv('../io/live_test/perf/SFStrategyI-17-05-2024-23-20-55SFStrategyI__0.001.csv')

plt.figure(figsize=(8, 5))
# Plot the data
plt.plot(data_1['iter'], data_1['wallet_tot'], label='SF 0.01')
plt.plot(data_2['iter'], data_2['wallet_tot'], label='SF 0.005')
plt.plot(data_3['iter'], data_3['wallet_tot'], label='SF 0.001')
plt.plot(data_4['iter'], data_4['wallet_tot'], label='SFI 0.01')
plt.plot(data_5['iter'], data_5['wallet_tot'], label='SFI 0.005')
plt.plot(data_6['iter'], data_6['wallet_tot'], label='SFI 0.001')

plt.title("Live test results over 48 hours")
plt.xlabel("Time (iterations)")
plt.ylabel("Total value (USD)")

plt.legend()
plt.savefig(f"../io/live_test/fig/live_test_48h_SFS_SFSI.pdf")

plt.show()
