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
data_1 = pd.read_csv('C:/Users/dadou/Documents/Github/Advanced-Programming/io/live_test/perf/SFStrategyI-19-04-2024-13-55-44SFStrategyI__0.0005.csv')
data_2 = pd.read_csv('C:/Users/dadou/Documents/Github/Advanced-Programming/io/live_test/perf/SFStrategyI-19-04-2024-13-56-24SFStrategyI__0.005.csv')
data_3 = pd.read_csv('C:/Users/dadou/Documents/Github/Advanced-Programming/io/live_test/perf/SFStrategyI-19-04-2024-13-56-07SFStrategyI__0.001.csv')

# Plot the data
plt.plot(data_1['iter'], data_1['pnl_pos_tot'], label='0.0005')
plt.plot(data_2['iter'], data_2['pnl_pos_tot'], label='0.005')
plt.plot(data_3['iter'], data_3['pnl_pos_tot'], label='0.001')

plt.legend()
plt.show()
