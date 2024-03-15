import plotly.express as px
import pandas as pd

# Load the data
data = pd.read_csv('../Data/crypto-1min/eth-min1/cleaned_data/ETH_MACD_march2024.csv')

# Keep only the last 4 hours
data = data.iloc[-60:]

# Create a figure
fig = px.line(data, x='date', y='close')

# Add the RSI to the figure with a secondary y-axis
fig.add_scatter(x=data['date'], y=data['RSI'], yaxis='y2')

# Update the layout
fig.update_layout(yaxis=dict(title='Price'), yaxis2=dict(title='RSI', overlaying='y', side='right'))

# Show the figure
fig.show()
