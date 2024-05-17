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
import csv

# laod env variables
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

env = os.getenv("environment")


class SFStrategy:

    def __init__(self, run_name, buy_percentage=0.01, exposure=2, debug=True):
        """
        Initialize the strategy

        :param run_name: Name of the run
        :param buy_percentage: The percentage range to create the orders
        ex. if buy_percentage = 0.01, the orders will be created at 1% intervals (200, 198, ...)
        :param exposure: Wallet exposure to the orders
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

        t_string = f"{time.day}/{time.month}/{time.year}-{time.hour}:{time.minute}:{time.second}"

        # Amount of index to buy per order
        am_ = wallet * self.exposure / 100 / data

        # If the order list and the position list are empty, create the orders
        # Create 100 orders
        if len(order_list) == 0 and len(position_list) == 0:
            for value in [data * (1 - (i * self.buy_percentage)) for i in range(1, 100)]:
                if self.debug:
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

                    if self.debug:
                        # Print, notification and log the order
                        print_green(emoji.emojize(":green_circle:") + t_string + f"Order condition met at {data}")
                        print_green(f"\t Position size: {order.amount} at {data}")
                        print_green(f"\t Position value: {order.amount * data}")
                        send_message(
                            emoji.emojize(":green_circle:") +
                            f"Order condition met at {data}\n",
                            f"Position size: {order.amount} at {data}\n"
                            f"Position value: {order.amount * data}\n"
                            f"Wallet: {wallet}",
                        )
                        if env == "server":
                            # Check if file exists
                            if not os.path.exists('../io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                                subprocess.run(
                                    ["touch", '../io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                                with open('../io/live_test/log/live_test_log_' + self.run_name + '.csv',
                                          'w') as file:
                                    file.write("time,"
                                               "event,"
                                               "event_price,"
                                               "position_size,"
                                               "position_value,"
                                               "wallet,"
                                               "infos"
                                               "\n")
                            # Log the order
                            with open('../io/live_test/log/live_test_log_' + self.run_name + '.csv', 'a') as file:
                                file.write(t_string + ",")
                                file.write(f"LONG ORDER,")
                                file.write(f"{data},")
                                file.write(f"{order.amount},")
                                file.write(f"{order.amount * data},")
                                file.write(f"{wallet},")
                                file.write(f"LONG Order condition met at {data}")
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

                if self.debug:
                    print_red(emoji.emojize(":red_circle:") + t_string + f"Position closing condition met at {data}")
                    print_red(f"\t Position size: {position.amount} at {data}")
                    print_red(f"\t Position value: {position.amount * data}")
                    send_message(
                        emoji.emojize(":red_circle:") +
                        f"Position closing condition met at {data}\n",
                        f"Position size: {position.amount} at {data}\n"
                        f"Position value: {position.amount * data}\n"
                        f"Wallet: {wallet}",
                    )
                    if env == "server":
                        # Check if file exists
                        if not os.path.exists('../io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                            subprocess.run(["touch", '../io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                            with open('../io/live_test/log/live_test_log_' + self.run_name + '.csv', 'w') as file:
                                file.write("time,"
                                           "event,"
                                           "event_price,"
                                           "position_size,"
                                           "position_value,"
                                           "wallet,"
                                           "infos"
                                           "\n")
                        # Log the order
                        with open('../io/live_test/log/live_test_log_' + self.run_name + '.csv',
                                  'a') as file:
                            file.write(t_string + ",")
                            file.write(f"LONG POSITION,")
                            file.write(f"{data},")
                            file.write(f"{position.amount},")
                            file.write(f"{position.amount * data},")
                            file.write(f"{wallet},")
                            file.write(f"LONG Position condition met at {data}")
                            file.write(f" -- From order created at {position.opening_time}\n")

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

    def __str__(self):
        return "SFStrategy-" + self.run_name
