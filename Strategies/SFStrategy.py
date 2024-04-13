import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm

from utils.market import *
from utils.utils import *


class SFStrategy:

    def __init__(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
            self.df = pd.read_csv(data_path)

            self.opening_time = self.df['date'].values[0]
            self.opening_price = self.df['close'].values[0]
            self.closing_time = self.df['date'].values[-1]
            self.closing_price = self.df['close'].values[-1]
            self.positions = []
            self.orders = []

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

    def check_conditions(self, orders, positions, data, time, wallet):
        """
        Central part of the strategy class. It takes as input the list of positions and orders and the data.
        The data is composed of the prices of the index as well as what the strategy needs to make a decision.
        In this case, the strategy is very simple and only needs the closing prices.
        :param orders: A list of the orders
        :param positions: A list of the positions
        :param data: The index price
        :param time: The time of the index price
        :param wallet: The amount disposable to trade
        :return: The update lists of orders and positions
        """
        order_list = orders
        position_list = positions
        wallet = wallet
        buy_percentage = 0.00025
        dollars_value = 0
        exposure = 0.5  # 50% of the wallet is open to create orders

        # Amount of index to buy per order
        am_ = wallet * exposure / 100 / data

        # If the order list and the position list are empty, create the orders
        # Create 100 orders
        if len(order_list) == 0 and len(position_list) == 0:
            for value in [data * (1-(i*buy_percentage)) for i in range(1, 100)]:

                print(f"Creating order at {value}")

                order_list.append(Order(
                    time=time,
                    price=value,
                    amount=am_,
                    direction='long'))

        # Check orders
        for order in order_list:
            # Case 1: The order is a long order and the index price is below the order price -> execute the order
            if order.direction == 'long' and data <= order.price:
                if wallet >= order.amount:

                    print_green(f"Order condition met at {data}, {time}")

                    # Create a new position
                    p = Position(
                            opening_price=data,
                            opening_time=time,
                            amount=am_,
                            direction="long",
                            closing_price=data * (1 + buy_percentage)
                    )

                    # Add the position to the position list
                    position_list.append(p)

                    # Update the wallet value
                    wallet -= order.amount * data

                    # Remove the order from the order list
                    del order_list[0]
                else:
                    print(f'Not enough money to buy {order.amount} at {data}')

        # Check positions
        for position in position_list:
            # Case 1: The position is a long position and the index price is above the closing price -> close the pos
            if position.direction == 'long' and data >= position.closing_price and position.status == 'open':

                print_red(f"Position closing condition met at {data}, {time}")

                position.close(data, time)
                dollars_value += position.dollars_value(data)

                # Reset the order book
                order_list = []
                for value in [data * (1-(i*buy_percentage)) for i in range(1, 100)]:
                    order_list.append(Order(
                        time=time,
                        price=value,
                        amount=am_,
                        direction='long'))

        # Update the wallet by adding profits made by closing positions
        wallet += dollars_value

        return order_list, position_list, wallet

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

        # For each time step check if orders should be executed or if positions should be closed
        for i in tqdm(range(1, len(ts))):

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

        nb_open_positions = 0
        for position in self.positions:
            if position.status == 'open':
                nb_open_positions += 1

            # Write the results to a file in io folder
        with open('io/results.txt', 'a') as file:

            file.write(f"SFStrategy: {self.data_path} \n")
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
