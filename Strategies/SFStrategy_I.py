"""
This file contains the SFStrategy class. This class is a simple straightforward strategy that improves the
SFStrategy. This one add the possibility to create orders to trigger short sell positions.
The order book is generated the same way as for the SFStrategy, for both directions.
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


class SFStrategyI:

    def __init__(self, run_name, data_path=None, buy_percentage=0.01, exposure=2):
        """
        Initialize the strategy

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
            self.positions_long = []
            self.positions_short = []
            self.orders_long = []
            self.orders_short = []

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

        # Create an order list for the long and short positions
        order_list_long = []
        order_list_short = []
        for orders in order_list:
            if orders.direction == 'long':
                order_list_long.append(orders)
            elif orders.direction == 'short':
                order_list_short.append(orders)

        # Create a position list for the long and short positions
        position_list_long = []
        position_list_short = []
        for positions in position_list:
            if positions.direction == 'long':
                position_list_long.append(positions)
            elif positions.direction == 'short':
                position_list_short.append(positions)

        t_string = f"{time.day}/{time.month}/{time.year}-{time.hour}:{time.minute}:{time.second}"

        # Amount of index to buy per order
        am_ = wallet * self.exposure / 100 / data

        # If the order list and the position list are empty, create the orders
        # Initial order book, create 50 orders in each direction
        if len(order_list) == 0 and len(position_list) == 0:
            for i in range(1, 50):
                long_open_value = data * (1 - (i * self.buy_percentage))
                short_open_value = data * (1 + (i * self.buy_percentage))
                print(f"Long: {long_open_value}, Short: {short_open_value}")

                order_list_long.append(Order(
                    time=time,
                    price=long_open_value,
                    amount=am_,
                    direction='long'))

                order_list_short.append(Order(
                    time=time,
                    price=short_open_value,
                    amount=am_,
                    direction='short'))

        # --------------------------------------------- Check orders --------------------------------------------- #
        # Case 1: The order is a long order and the index price is below the order price -> execute the order
        for order in order_list_long:
            if order.direction == 'long' and data <= order.price:
                if wallet >= order.amount:

                    # Print, notification and log the order
                    print_green(emoji.emojize(":green_circle:") + t_string + f"LONG order condition met at {data}")
                    print_green(f"\t Position size: {order.amount} at {data}")
                    print_green(f"\t Position value: {order.amount * data}")
                    send_message(
                        emoji.emojize(":green_circle:") +
                        f"LONG Order condition met at {round(data, 3)}\n",
                        f"Position size: {round(order.amount, 3)} at {round(data, 3)}\n"
                        f"Position value: {round(order.amount * data, 3)}\n"
                        f"Wallet: {round(wallet, 5)}",
                    )
                    if env == "server":
                        # Check if file exists
                        if not os.path.exists('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt'):
                            with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'w') as file:
                                file.write("Log file for the live test of the SFStrategyI\n\n\n")
                        # Log the order
                        with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'a') as file:
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
                    position_list_long.append(p)

                    # Update the wallet value
                    wallet -= order.amount * data

                    # Remove the order from the order list
                    del order_list_long[0]
                else:
                    print(f'Not enough money to buy {order.amount} at {data}')

        # Case 2: The order is a short order and the index price is over the order price -> execute the order
        for order in order_list_short:

            if order.direction == 'short' and data >= order.price:
                # No conditions on the wallet to open a short position

                # Print, notification and log the order
                print_green(emoji.emojize(":green_circle:") + t_string + f"SHORT order condition met at {data}")
                print_green(f"\t Position size: {order.amount} at {data}")
                print_green(f"\t Position value: {order.amount * data}")
                send_message(
                    emoji.emojize(":green_circle:") +
                    f"SHORT Order condition met at {round(data, 3)}\n",
                    f"Position size: {round(order.amount, 3)} at {round(data, 3)}\n"
                    f"Position value: {round(order.amount * data, 3)}\n"
                    f"Wallet: {round(wallet, 5)}",
                )
                if env == "server":
                    # Check if file exists
                    if not os.path.exists('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt'):
                        with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'w') as file:
                            file.write("Log file for the live test of the SFStrategyI\n\n\n")
                    # Log the order
                    with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'a') as file:
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
                    direction="short",
                    closing_price=data * (1 - self.buy_percentage)
                )

                # Add the position to the position list
                position_list_short.append(p)

                # Update the wallet value
                wallet += order.amount * data

                # Remove the order from the order list
                del order_list_short[0]

        # -------------------------------------------- Check positions -------------------------------------------- #
        # Case 1: The position is a long position and the index price is above the closing price -> close the pos
        for position in position_list_long:
            if position.direction == 'long' and data >= position.closing_price and position.status == 'open':

                print_red(emoji.emojize(":red_circle:") + t_string + f"LONG position closing condition met at {data}")
                print_red(f"\t Position size: {position.amount} at {data}")
                print_red(f"\t Position value: {position.amount * data}")
                send_message(
                    emoji.emojize(":red_circle:") +
                    f"LONG Position closing condition met at {round(data, 3)}\n",
                    f"Position size: {round(position.amount, 3)} at {round(data, 3)}\n"
                    f"Position value: {round(position.amount * data, 3)}\n"
                    f"Wallet: {round(wallet, 5)}",
                )
                if env == "server":
                    # Check if file exists
                    if not os.path.exists('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt'):
                        with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'w') as file:
                            file.write("Log file for the live test of the SFStrategyI\n\n\n")
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
                order_list_long = []
                order_list_short = []
                for i in range(1, 50):
                    long_open_value = data * (1 - (i * self.buy_percentage))
                    short_open_value = data * (1 + (i * self.buy_percentage))
                    amount_ = wallet * self.exposure / 100 / data

                    order_list_long.append(Order(
                        time=time,
                        price=long_open_value,
                        amount=amount_,
                        direction='long'))

                    order_list_short.append(Order(
                        time=time,
                        price=short_open_value,
                        amount=amount_,
                        direction='short'))

        # Case 2: The position is a short position and the index price is below the closing price -> close the pos
        for position in position_list_short:
            if position.direction == 'short' and data <= position.closing_price and position.status == 'open':

                print_red(
                    emoji.emojize(":red_circle:") + t_string + f"SHORT position closing condition met at {data}")
                print_red(f"\t Position size: {position.amount} at {data}")
                print_red(f"\t Position value: {position.amount * data}")
                send_message(
                    emoji.emojize(":red_circle:") +
                    f"SHORT Position closing condition met at {round(data, 3)}\n",
                    f"Position size: {round(position.amount, 3)} at {round(data, 3)}\n"
                    f"Position value: {round(position.amount * data, 3)}\n"
                    f"Wallet: {round(wallet, 5)}",
                )
                if env == "server":
                    # Check if file exists
                    if not os.path.exists('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt'):
                        with open('io/live_test/log/live_test_log_SFStrategyI' + self.run_name + '.txt', 'w') as file:
                            file.write("Log file for the live test of the SFStrategyI\n\n\n")
                    # Log the order
                    with open('io/live_test/live_test_log_SFStrategy' + self.run_name + '.txt', 'a') as file:
                        file.write(t_string + "\n")
                        file.write(f"Position closing condition met at {round(data, 3)}\n")
                        file.write(f"Position size: {round(position.amount, 3)} at {round(data, 3)}\n")
                        file.write(f"Position value: {round(position.amount * data, 3)}\n")
                        file.write(f"Wallet: {round(wallet, 5)}\n\n\n")

                position.close(data, time)
                dollars_value += position.dollars_value(data)

                # Reset the order book
                order_list_long = []
                order_list_short = []
                for i in range(1, 50):
                    long_open_value = data * (1 - (i * self.buy_percentage))
                    short_open_value = data * (1 + (i * self.buy_percentage))
                    amount_ = wallet * self.exposure / 100 / data

                    order_list_long.append(Order(
                        time=time,
                        price=long_open_value,
                        amount=amount_,
                        direction='long'))

                    order_list_short.append(Order(
                        time=time,
                        price=short_open_value,
                        amount=amount_,
                        direction='short'))

        # Update the wallet by adding profits made by closing positions
        wallet += dollars_value

        order_list = order_list_long + order_list_short
        position_list = position_list_long + position_list_short

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
        return "SFStrategyI-" + self.run_name