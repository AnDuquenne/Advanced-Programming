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
import subprocess
from dotenv import load_dotenv

load_dotenv()

env = os.getenv("environment")


class SFStrategyI:

    def __init__(self, run_name, data_path=None, buy_percentage=0.01, exposure=2, debug=True):
        """
        Initialize the strategy

        :param data_path: Path to the data for the backtest
        :param buy_percentage: The percentage range to create the orders
        ex. if buy_percentage = 0.01, the orders will be created at 1% intervals (200, 198, ...)
        :param exposure: Wallet exposure to the orders
        :param debug: debug parameter
        ex. if exposure = 0.5, 50% of the wallet is open to create the order book
        """

        self.run_name = run_name
        self.buy_percentage = buy_percentage
        self.exposure = exposure  # x % of the wallet is open to create orders
        self.debug = debug

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
                if self.debug:
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

                    if self.debug:
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
                            if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                                subprocess.run(["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                                with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'w') as file:
                                    file.write("time,"
                                               "event,"
                                               "event_price,"
                                               "position_size,"
                                               "position_value,"
                                               "wallet,"
                                               "infos"
                                               "\n")
                            # Log the order
                            with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'a') as file:
                                file.write(t_string + ",")
                                file.write(f"LONG ORDER,")
                                file.write(f"{round(data, 3)},")
                                file.write(f"{round(order.amount, 3)},")
                                file.write(f"{round(order.amount * data, 3)},")
                                file.write(f"{round(wallet, 5)},")
                                file.write(f"LONG Order condition met at {round(data, 3)}")
                                file.write(f" -- From order created at {order.time}\n")

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

                if self.debug:
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
                        if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                            subprocess.run(["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                            with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'w') as file:
                                file.write("time,"
                                           "event,"
                                           "event_price,"
                                           "position_size,"
                                           "position_value,"
                                           "wallet,"
                                           "infos"
                                           "\n")
                        # Log the order
                        with open('io/live_test/log/live_test_log_' + self.run_name + '.csv',
                                  'a') as file:
                            file.write(t_string + ",")
                            file.write(f"SHORT ORDER,")
                            file.write(f"{round(data, 3)},")
                            file.write(f"{round(order.amount, 3)},")
                            file.write(f"{round(order.amount * data, 3)},")
                            file.write(f"{round(wallet, 5)},")
                            file.write(f"SHORT Order condition met at {round(data, 3)}")
                            file.write(f" -- From order created at {order.time}\n")

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

                if self.debug:
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
                        if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                            subprocess.run(["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                            with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'w') as file:
                                file.write("time,"
                                           "event,"
                                           "event_price,"
                                           "position_size,"
                                           "position_value,"
                                           "wallet,"
                                           "infos"
                                           "\n")
                        # Log the order
                        with open('io/live_test/log/live_test_log_' + self.run_name + '.csv',
                                  'a') as file:
                            file.write(t_string + ",")
                            file.write(f"LONG POSITION,")
                            file.write(f"{round(data, 3)},")
                            file.write(f"{round(position.amount, 3)},")
                            file.write(f"{round(position.amount * data, 3)},")
                            file.write(f"{round(wallet, 5)},")
                            file.write(f"LONG Position condition met at {round(data, 3)}")
                            file.write(f" -- From order created at {position.opening_time}\n")

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

                if self.debug:
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
                        if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                            subprocess.run(["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                            with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'w') as file:
                                file.write("time,"
                                           "event,"
                                           "event_price,"
                                           "position_size,"
                                           "position_value,"
                                           "wallet,"
                                           "infos"
                                           "\n")
                        # Log the order
                        with open('io/live_test/log/live_test_log_' + self.run_name + '.csv',
                                  'a') as file:
                            file.write(t_string + ",")
                            file.write(f"SHORT POSITION,")
                            file.write(f"{round(data, 3)},")
                            file.write(f"{round(position.amount, 3)},")
                            file.write(f"{round(position.amount * data, 3)},")
                            file.write(f"{round(wallet, 5)},")
                            file.write(f"SHORT Position condition met at {round(data, 3)}")
                            file.write(f" -- From order created at {position.opening_time}\n")

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

    def __str__(self):
        return "SFStrategyI-" + self.run_name