import pandas as pd

file = 'Data/crypto-1min/eth-min1/ETH_MACD_march2024'
# Load the data
df = pd.read_excel(file + '.xlsx')

#remove the first row
df = df.iloc[1:]

# Rename the columns
df.columns = ['date', 'close', 'macd_histogram', 'macd_macd', 'macd_signal']

# A code to revers the order of the dataframe
df = df.iloc[::-1]
df.to_csv(file + '.csv', index=False)
