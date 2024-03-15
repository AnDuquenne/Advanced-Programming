import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm

from utils.market import *


class SFStrategyI:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.positions = []
        self.orders = []
        self.opening_time = self.df['date'].values[0]
        self.opening_price = self.df['close'].values[0]
        self.closing_time = self.df['date'].values[-1]
        self.closing_price = self.df['close'].values[-1]

        self.wallet = 10000

    def analyze_chart(self):

        # make a plotly chart of the data with a small horizontal line for each 1% away from the opening price
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=self.df['date'], y=self.df['close'], mode='lines', name='Close price'), row=1, col=1)

        fig.add_trace(go.Scatter(x=[self.df['date'].values[0], self.df['date'].values[-1]],
                                 y=[self.df['close'].values[0], self.df['close'].values[-1]],
                                 mode='lines', name='Opening price'), row=1, col=1)

        for value in np.linspace(self.df['close'].values[0], 0, 101)[1:]:
            fig.add_trace(go.Scatter
                          (x=[self.df['date'].values[0], self.df['date'].values[-1]], y=[value, value],
                           mode='lines', name='Opening price'), row=1, col=1)

        fig.show()

    def run(self):

        ts = self.df['close']
        initial_value = ts[0]

        buy_percentage = 0.01

        profits = np.zeros(len(ts))

        # Create orders
        for value in np.linspace(self.df['close'].values[0], 0, 101)[1:]:
            self.orders.append(Order(
                time=self.df["date"].values[0],
                price=value,
                amount=100,
                direction='long'))

        no_profit_period = 0

        # For each time step check if orders should be executed or if positions should be closed
        for i in tqdm(range(1, len(ts))):

            # If no profit for 1 day, reset order book
            if no_profit_period > 1440:
                self.orders = []
                for value in np.linspace(ts[i], 0, 101)[1:]:
                    self.orders.append(Order(
                        time=self.df["date"].values[0],
                        price=value,
                        amount=100,
                        direction='long'))
                no_profit_period = 0

            profit = 0

            # Check orders
            for order in self.orders:
                if order.direction == 'long' and ts[i] <= order.price:
                    if self.wallet >= order.amount:

                        self.positions.append(
                            Position(
                                opening_price=ts[i],
                                opening_time=self.df['date'].values[i],
                                amount=order.amount,
                                direction="long",
                                closing_price=ts[i] + (ts[i] * buy_percentage))
                        )
                        self.wallet -= order.amount
                        del self.orders[0]
                    else:
                        print(f'Not enough money to buy {order.amount} at {ts[i]}')

                    print(f"Order condition met at {ts[i]}, {i}")
                    print(f'Wallet: {self.wallet}')
                    print(f'Orders: {len(self.orders)}')
                    print(f'Positions: {len(self.positions)}')

            # Check positions
            for position in self.positions:
                if position.direction == 'long' and ts[i] >= position.closing_price and position.status == 'open':

                    profit += position.close(ts[i], self.df['date'].values[i])

                    self.wallet += profit

                    # Reset the order book
                    self.orders = []
                    for value in np.linspace(ts[i], 0, 101)[1:]:
                        self.orders.append(Order(
                            time=self.df["date"].values[0],
                            price=value,
                            amount=100,
                            direction='long'))

                    print(f"Position closing condition met at {ts[i]}, {i}")
                    print(f'Wallet: {self.wallet}')
                    print(f'Orders: {len(self.orders)}')
                    print(f'Positions: {len(self.positions)}')

        profits[i] = profit

        if len(self.positions) == 0:
            no_profit_period += 1

        nb_open_positions = 0
        for position in self.positions:
            if position.status == 'open':
                nb_open_positions += 1

        # Write the results to a file in io folder
        with open('io/results.txt', 'a') as file:

            file.write(f"SFStrategyI: {self.data_path} \n")
            file.write(f"Wallet: {self.wallet} \n")
            file.write(f"Orders: {len(self.orders)} \n")
            file.write(f"Nb open positions: {nb_open_positions} \n")

        return profits

    def show_simulation(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.df['date'], y=self.df['close'], mode='lines', name='Close price'))

        # For each positions, add a dot on the graph, a green dot at the opening price and a red dot at the closing price
        for position in self.positions:
            fig.add_trace(go.Scatter(x=[position.opening_time, position.closing_time],
                                     y=[position.opening_price, position.closing_price],
                                     mode='markers', name='Position', marker=dict(color='green')))
            fig.add_trace(go.Scatter(x=[position.closing_time],
                                     y=[position.closing_price],
                                     mode='markers', name='Position', marker=dict(color='red')))

        fig.show()


