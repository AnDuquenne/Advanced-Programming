"""
This file contains the StrategyMACD class.
When the MACD signal is triggered, the strategy opens a long position.
The long position is closed when the index price is above the closing price.
The closing price is the price at which the position was opened plus a buy_percentage.
The MACD signal is triggered when the histogram value is below 0 and greater than the one of the previous period.
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
import os, sys
import subprocess
from dotenv import load_dotenv

load_dotenv()

env = os.getenv("environment")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.technical_analysis import MACD


class StrategyMACD:

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
        # For back testing
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

        self.run_name = run_name
        self.buy_percentage = buy_percentage
        self.exposure = exposure  # x % of the wallet is open to create orders

        self.last_macd_hist_value = None

    def get_macd_histogram(self, history):
        """
        Get the historical MACD values for the last period.
        :return:
        """
        # Get the historical price
        close_prices = history

        # Compute the MACD
        macd = MACD(close_prices)
        return macd.get_histogram()

    def get_macd_trigger(self, history):
        """
        Get the MACD trigger value.
        :return: True if the MACD signal is triggered
        """
        histogram = self.get_macd_histogram(history)
        if histogram[-3] > histogram[-2] and 0 >= histogram[-1] > histogram[-2]:
            return True
        else:
            return False

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

        history = data['history']
        data = data['price']

        t_string = f"{time.day}/{time.month}/{time.year}-{time.hour}:{time.minute}:{time.second}"

        # Amount of index to buy per order
        am_ = wallet * self.exposure / 50 / data

        # ------------------------------------ Check conditions to open a pos ------------------------------------ #

        # Check if the MACD signal is triggered
        trigger = self.get_macd_trigger(history)

        if trigger and self.last_macd_hist_value != history[-2] and wallet >= am_ * data:
            # If the MACD signal is triggered, open a position
            position_list.append(Position(
                opening_time=time,
                opening_price=data,
                amount=am_,
                direction='long',
                closing_price=data * (1 + self.buy_percentage)
            ))

            wallet -= am_ * data

            # Reset the last MACD histogram value
            self.last_macd_hist_value = history[-2]

            # Print, notification and log the order
            print_green(emoji.emojize(":green_circle:") + t_string + f"Position opening condition met at {data}")
            print_green(f"\t Position size: {am_} at {data}")
            print_green(f"\t Position value: {am_ * data}")
            send_message(
                emoji.emojize(":green_circle:") +
                f"Order condition met at {round(data, 3)}\n",
                f"Position size: {round(am_, 3)} at {round(data, 3)}\n"
                f"Position value: {round(am_ * data, 3)}\n"
                f"Wallet: {round(wallet, 5)}",
            )
            if env == "server":
                # Check if file exists
                if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                    subprocess.run(
                        ["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                    with open('io/live_test/log/live_test_log_' + self.run_name + '.csv',
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
                with open('io/live_test/log/live_test_log_' + self.run_name + '.csv', 'a') as file:
                    file.write(t_string + ",")
                    file.write(f"LONG ORDER,")
                    file.write(f"{round(data, 3)},")
                    file.write(f"{round(am_, 3)},")
                    file.write(f"{round(am_ * data, 3)},")
                    file.write(f"{round(wallet, 5)},")
                    file.write(f"Open position met\n")

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
                    if not os.path.exists('io/live_test/log/live_test_log_' + self.run_name + '.csv'):
                        subprocess.run(
                            ["touch", 'io/live_test/log/live_test_log_' + self.run_name + '.csv'])
                        with open('io/live_test/log/live_test_log_' + self.run_name + '.csv',
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

        # Update the wallet by adding profits made by closing positions
        wallet += dollars_value

        return order_list, position_list, wallet

    def __str__(self):
        return "Strategy_MACD-" + self.run_name
