import asyncio
import pandas as pd

import numpy as np
import websockets
import requests
import time
import datetime
import json

import emoji

import sys
import os
import subprocess

# load environment variables
from dotenv import load_dotenv

load_dotenv()
env = os.getenv("environment")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import *
from utils.technical_analysis import MACD
from utils.notifications import send_message

from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_FC import StrategyFC
from Strategies.Strategy_LSTM import StrategyLSTM
from Strategies.Strategy_MACD import StrategyMACD


class LiveTest():
    def __init__(self, index, strategy, wallet):
        self.index = index

        self.orders = []
        self.positions = []
        self.wallet = wallet

        self.strategy = strategy

    async def call_api(self, msg):
        """
        Call the deribit API

        :param msg: request message
        :return:
        """
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                response = json.loads(response)
                index_price = response["result"]["index_price"]
                return index_price

    def get_price(self):
        """
        Get the index price at the current time.

        This method uses the API provided by www.deribit.com as this was the chosen platform to perform the live test.
        :return:
        """
        msg = \
            {"jsonrpc": "2.0",
             "method": "public/get_index_price",
             "id": 42,
             "params": {
                 "index_name": self.index}
             }
        price = asyncio.get_event_loop().run_until_complete(self.call_api(json.dumps(msg)))
        return price, datetime.datetime.now()

    def get_historical_price(self, days: int, minutes: int, interval: str = "5"):
        """
        Get the historical price of the index for the last x days + y minutes.

        This method uses the API provided by www.bybit.com, as deribit api fails to provide correct data.
        https://bybit-exchange.github.io/docs/api-explorer/v5/market/mark-kline
            Response Parameters
            -------------------
            Parameter	Type	Comments
            category	string	Product type
            symbol	string	Symbol name
            list	array
            An string array of individual candle
            Sort in reverse by startTime
            > list[0]: startTime	string	Start time of the candle (ms)
            > list[1]: openPrice	string	Open price
            > list[2]: highPrice	string	Highest price
            > list[3]: lowPrice	string	Lowest price
            > list[4]: closePrice	string	Close price. Is the last traded price when the candle is not closed
        :param days: number of days to get the historical price
        :param minutes: number of minutes to get the historical price
        :param interval: interval minutes of the candles
        :return: list of close prices for the granularity of the interval
        """
        # Get the current timestamp and the timestamp of the last x days
        t0, t1 = get_timestamps(datetime.timedelta(days=days, minutes=minutes))

        url = "https://api-testnet.bybit.com/v5/market/mark-price-kline"

        # Index does not have an underscore on bybit, and are written in uppercase
        bybit_index = self.index.upper().replace("_", "")

        params = {
            "symbol": bybit_index,
            "interval": interval,
            "start": t0,
            "end": t1,
            "limit": 1000
        }

        response = requests.request("GET", url, params=params)
        response = json.loads(response.text)["result"]["list"]  # list of candles (see above)

        # first element is the most recent candle, we reverse the list
        response = response[::-1]

        close_prices = [float(candle[4]) for candle in response]

        return close_prices

    def run(self):
        """
        Runs a live test
        :return:
        """

        waiting_time = 0

        notif_sent = False
        iter_log = 0

        # Every x seconds get the index price
        while True:

            price, t = self.get_price()

            if isinstance(self.strategy, StrategyMACD):
                try:
                    history = self.get_historical_price(days=0, minutes=60, interval="5")
                    data = {
                        "price": price,
                        "history": history
                    }
                    waiting_time = 1
                    self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                        orders=self.orders,
                        positions=self.positions,
                        data=data,
                        time=t,
                        wallet=self.wallet,
                    )
                except:
                    print_red("Error getting historical data")
                    waiting_time = 1
                    continue
            elif isinstance(self.strategy, StrategyFC):
                try:
                    data = self.get_historical_price(days=0, minutes=10, interval="1")
                    # Transform price to a pandas series
                    data = pd.Series(data)
                    waiting_time = 59
                    self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                        orders=self.orders,
                        positions=self.positions,
                        data=data,
                        time=t,
                        wallet=self.wallet,
                    )
                except:
                    print_red("Error getting historical data")
                    waiting_time = 1
                    continue
            elif isinstance(self.strategy, StrategyLSTM):
                try:
                    data = self.get_historical_price(days=0, minutes=10, interval="1")
                    # Transform price to a pandas series
                    data = pd.Series(data)
                    waiting_time = 59
                    self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                        orders=self.orders,
                        positions=self.positions,
                        data=data,
                        time=t,
                        wallet=self.wallet,
                    )
                except:
                    print_red("Error getting historical data")
                    waiting_time = 1
                    continue

            else:
                try:
                    self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                        orders=self.orders,
                        positions=self.positions,
                        data=price,
                        time=t,
                        wallet=self.wallet,
                    )
                    waiting_time = 0.5
                except:
                    print_red("Error getting historical data")
                    waiting_time = 1
                    continue

            t_string = f"{t.day}/{t.month}/{t.year}-{t.hour}:{t.minute}:{t.second}"

            if env == "local":
                print_yellow(f"------- Update at " + t_string + " -------")



            # ----------------- Print the results ----------------- #
            if env == "local":
                print(f"Wallet: {self.wallet}")
                print(f"Price: {price}")

            # Positions
            open_pos_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    open_pos_ += 1

            if env == "local":
                print(f"Positions:")
                print("      open: ", open_pos_)
                print("      closed: ", len(self.positions) - open_pos_)
                print("      total: ", len(self.positions))

            # PNLs
            # PNL of closed pos
            pnl_closed = 0
            pnl_open = 0
            for pos_ in self.positions:
                if pos_.status == 'closed':
                    pnl_closed += pos_.profit
                elif pos_.status == 'open':
                    pnl_open += pos_.pnl(price)

            if env == "local":
                print(f"PNL of positions")
                print("       Closed: ", pnl_closed)
                print("       Open: ", pnl_open)
                print("       Total: ", pnl_closed + pnl_open)

            # Total value of positions
            tot_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    tot_ += pos_.dollars_value(price)

            if env == "local":
                print(f"Total value of positions: {tot_}")
                print(f"Total value of the wallet (pos+wallet): {tot_ + self.wallet}")

            # To avoid sending multiple notifications inside the same second
            if t.second == 0:
                notif_sent = False

            # Duplicate code because of different timing for networks
            if (isinstance(self.strategy, SFStrategy) or
                    isinstance(self.strategy, SFStrategyI) or
                    isinstance(self.strategy, StrategyMACD)):
                if t.second == 30 and t.minute % 2 == 0 and not notif_sent:
                    title = emoji.emojize(":eight-thirty:") + " Recurrent update " + emoji.emojize(":eight-thirty:") + "\n\n"
                    msg = ""
                    msg += f"Positions:\n"
                    msg += f"\t\t open: {open_pos_}\n"
                    msg += f"\t\t closed: {len(self.positions) - open_pos_}\n"
                    msg += f"\t\t total: {len(self.positions)}\n"
                    msg += f"PNL of positions\n"
                    msg += f"\t\t Closed: {pnl_closed}\n"
                    msg += f"\t\t Open: {pnl_open}\n"
                    msg += f"\t\t Total: {pnl_closed + pnl_open}\n"
                    msg += f"Total value of positions: {tot_}\n"
                    msg += f"Total value of the wallet (pos+wallet): {tot_ + self.wallet}\n"
                    send_message(title, msg)
                    notif_sent = True
                    iter_log += 1

                    if env == "server":
                        # Record pnl and wallet each x minutes
                        # Check if file exists
                        if not os.path.exists("../io/live_test/perf/" + self.strategy.__str__() + ".csv"):
                            subprocess.run(["touch", "../io/live_test/perf/" + self.strategy.__str__() + ".csv"])
                            with open("../io/live_test/perf/" + self.strategy.__str__() + ".csv", "w") as f:
                                f.write("iter,time,open_pos,closed_pos,tot_pos,"
                                        "pnl_pos_closed,pnl_pos_open,pnl_pos_tot,wallet_tot\n")

                        # Write the data
                        with open("../io/live_test/perf/" + self.strategy.__str__() + ".csv", "a") as f:
                            f.write(f"{iter_log},"
                                    f"{t},"
                                    f"{open_pos_},"
                                    f"{len(self.positions) - open_pos_},"
                                    f"{len(self.positions)},"
                                    f"{pnl_closed},"
                                    f"{pnl_open},"
                                    f"{pnl_closed + pnl_open},"
                                    f"{tot_ + self.wallet}\n")

            elif isinstance(self.strategy, StrategyFC) or isinstance(self.strategy, StrategyLSTM):
                if t.minute % 2 == 0 and not notif_sent:
                    title = emoji.emojize(":eight-thirty:") + " Recurrent update " + emoji.emojize(":eight-thirty:") + "\n\n"
                    msg = ""
                    msg += f"Positions:\n"
                    msg += f"\t\t open: {open_pos_}\n"
                    msg += f"\t\t closed: {len(self.positions) - open_pos_}\n"
                    msg += f"\t\t total: {len(self.positions)}\n"
                    msg += f"PNL of positions\n"
                    msg += f"\t\t Closed: {pnl_closed}\n"
                    msg += f"\t\t Open: {pnl_open}\n"
                    msg += f"\t\t Total: {pnl_closed + pnl_open}\n"
                    msg += f"Total value of positions: {tot_}\n"
                    msg += f"Total value of the wallet (pos+wallet): {tot_ + self.wallet}\n"
                    send_message(title, msg)
                    iter_log += 1

                    if env == "server":
                        # Record pnl and wallet each x minutes
                        # Check if file exists
                        if not os.path.exists("../io/live_test/perf/" + self.strategy.__str__() + ".csv"):
                            subprocess.run(["touch", "../io/live_test/perf/" + self.strategy.__str__() + ".csv"])
                            with open("../io/live_test/perf/" + self.strategy.__str__() + ".csv", "w") as f:
                                f.write("iter,time,open_pos,closed_pos,tot_pos,"
                                        "pnl_pos_closed,pnl_pos_open,pnl_pos_tot,wallet_tot\n")

                        # Write the data
                        with open("../io/live_test/perf/" + self.strategy.__str__() + ".csv", "a") as f:
                            f.write(f"{iter_log},"
                                    f"{t},"
                                    f"{open_pos_},"
                                    f"{len(self.positions) - open_pos_},"
                                    f"{len(self.positions)},"
                                    f"{pnl_closed},"
                                    f"{pnl_open},"
                                    f"{pnl_closed + pnl_open},"
                                    f"{tot_ + self.wallet}\n")

            if env == "local":
                time.sleep(5 + waiting_time)
            else:
                time.sleep(waiting_time)
