import asyncio
import websockets
import requests
import time
import datetime
import json

import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import *
from utils.notifications import send_message

class LiveTest():
    def __init__(self, index, strategy, wallet):
        self.index = index

        self.orders = []
        self.positions = []
        self.wallet = wallet

        self.strategy = strategy

    async def call_api(self, msg):
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                response = json.loads(response)
                index_price = response["result"]["index_price"]
                return index_price

    def get_price(self):
        msg = \
            {"jsonrpc": "2.0",
             "method": "public/get_index_price",
             "id": 42,
             "params": {
                 "index_name": self.index}
             }
        price = asyncio.get_event_loop().run_until_complete(self.call_api(json.dumps(msg)))
        return price, datetime.datetime.now()

    def run(self):
        # Every 5 seconds get the index price
        while True:

            price, t = self.get_price()
            t_string = f"{t.day}/{t.month}/{t.year}-{t.hour}:{t.minute}:{t.second}"

            print_yellow(f"------- Update at " + t_string + " -------")

            self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                orders=self.orders,
                positions=self.positions,
                data=price,
                time=t,
                wallet=self.wallet
            )

            # ----------------- Print the results ----------------- #
            print(f"Wallet: {self.wallet}")
            print(f"Price: {price}")
            print(f"Orders[0]: {self.orders[0]}")

            # Positions
            open_pos_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    open_pos_ += 1
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

            print(f"PNL of positions")
            print("       Closed: ", pnl_closed)
            print("       Open: ", pnl_open)
            print("       Total: ", pnl_closed + pnl_open)

            # Toral value of positions
            tot_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    tot_ += pos_.dollars_value(price)

            print(f"Total value of positions: {tot_}")
            print(f"Total value of the wallet (pos+wallet): {tot_ + self.wallet}")

            if t.second == 30:
                title = "Recurrent update\n\n"
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

            time.sleep(1)
