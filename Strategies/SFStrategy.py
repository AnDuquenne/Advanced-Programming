"""
This file contains the SFStrategy class. This class is a simple straightforward strategy.
It creates 100 orders at the opening price of the index. Each order is x% below the previous one.
When the index price reaches an order price, the order is executed and a position is created.
The position is closed when the index price reaches the closing price of the position.
When a position is closed, the order book is replaced using the last price of the closing position as reference.
"""

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tqdm import tqdm

from utils.market import *
from utils.utils import *

from utils.notifications import send_message

import emoji

# laod env variables
import os
from dotenv import load_dotenv

load_dotenv()

env = os.getenv("environment")


class SFStrategy:

    def __init__(self, run_name, data_path=None, buy_percentage=0.01, exposure=2):
        """
        Initialize the strategy

        :param run_name: Name of the run
        :param data_path: Path to the data for the backtest
        :param buy_percentage: The percentage range to create the orders
        ex. if buy_percentage = 0.01, the orders will be created at 1% intervals (200, 198, ...)
        :param exposure: Wallet exposure to the orders
        ex. if exposure = 0.5, 50% of the wallet is open to create the order book
        """
        if data_path is not None:
            self.data_path = data_path
            self.df = pd.read_csv(data_path)

            self.opening_time = self.df['date'].values[0]
            self.opening_price = self.df['close'].values[0]
            self.closing_time = self.df['date'].values[-1]
            self.closing_price = self.df['close'].values[-1]
            self.positions = []
            self.orders = []

            self.wallet = 10000  # For backtest only

        self.run_name = run_name
        self.buy_percentage = buy_percentage
        self.exposure = exposure  # x % of the wallet is open to create orders

    def analyze_chart(self):
        """
        EDA of the data
        :return:
        """

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
        dollars_value = 0

        t_string = f"{time.day}/{time.month}/{time.year}-{time.hour}:{time.minute}:{time.second}"

        # Amount of index to buy per order
        am_ = wallet * self.exposure / 100 / data

        # If the order list and the position list are empty, create the orders
        # Create 100 orders
        if len(order_list) == 0 and len(position_list) == 0:
            for value in [data * (1 - (i * self.buy_percentage)) for i in range(1, 100)]:
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

                    # Print, notification and log the order
                    print_green(emoji.emojize(":green_circle:") + t_string + f"Order condition met at {data}")
                    print_green(f"\t Position size: {order.amount} at {data}")
                    print_green(f"\t Position value: {order.amount * data}")
                    send_message(
                        emoji.emojize(":green_circle:") +
                        f"Order condition met at {round(data, 3)}\n",
                        f"Position size: {round(order.amount, 3)} at {round(data, 3)}\n"
                        f"Position value: {round(order.amount * data, 3)}\n"
                        f"Wallet: {round(wallet, 5)}",
                    )
                    if env == "server":
                        # Check if file exists
                        if not os.path.exists('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt'):
                            # Create the file
                            with open('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt', 'w') as file:
                                file.write("Log of the SFStrategy\n\n")
                        # Log the order
                        with open('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt', 'a') as file:
                            file.write(t_string + "\n")
                            file.write(f"Order condition met at {round(data, 3)}\n")
                            file.write(f"Position size: {round(order.amount, 3)} at {round(data, 3)}\n")
                            file.write(f"Position value: {round(order.amount * data, 3)}\n")
                            file.write(f"Wallet: {round(wallet, 5)}\n\n\n")

                    # Create a new position
                    p = Position(
                        opening_price=data,
                        opening_time=time,
                        amount=order.amount,
                        direction="long",
                        closing_price=data * (1 + self.buy_percentage)
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

                print_red(emoji.emojize(":red_circle:") + t_string + f"Position closing condition met at {data}")
                print_red(f"\t Position size: {position.amount} at {data}")
                print_red(f"\t Position value: {position.amount * data}")
                send_message(
                    emoji.emojize(":red_circle:") +
                    f"Position closing condition met at {round(data, 3)}\n",
                    f"Position size: {round(position.amount, 3)} at {round(data, 3)}\n"
                    f"Position value: {round(position.amount * data, 3)}\n"
                    f"Wallet: {round(wallet, 5)}",
                )
                if env == "server":
                    # Check if file exists
                    if not os.path.exists('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt'):
                        # Create the file
                        with open('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt', 'w') as file:
                            file.write("Log of the SFStrategy\n\n")
                    # Log the order
                    with open('io/live_test/log/live_test_log_SFStrategy' + self.run_name + '.txt', 'a') as file:
                        file.write(t_string + "\n")
                        file.write(f"Position closing condition met at {round(data, 3)}\n")
                        file.write(f"Position size: {round(position.amount, 3)} at {round(data, 3)}\n")
                        file.write(f"Position value: {round(position.amount * data, 3)}\n")
                        file.write(f"Wallet: {round(wallet, 5)}\n\n\n")

                position.close(data, time)
                dollars_value += position.dollars_value(data)

                # Reset the order book
                order_list = []
                for value in [data * (1 - (i * self.buy_percentage)) for i in range(1, 100)]:
                    order_list.append(Order(
                        time=time,
                        price=value,
                        amount=position.amount,
                        direction='long'))

        # Update the wallet by adding profits made by closing positions
        wallet += dollars_value

        return order_list, position_list, wallet

    def run(self):
        """
        Used to perform backtest on the data recovered from the cedif and kaggle.
        :return:
        """

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
        """
        Plot the simulation for the backtesting of the strategy
        :return:
        """
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

    def __str__(self):
        return "SFStrategy-" + self.run_name