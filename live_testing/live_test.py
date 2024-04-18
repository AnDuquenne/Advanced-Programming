import asyncio
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

        notif_sent = False

        # Every x seconds get the index price
        while True:

            price, t = self.get_price()
            t_string = f"{t.day}/{t.month}/{t.year}-{t.hour}:{t.minute}:{t.second}"

            if env == "local":
                print_yellow(f"------- Update at " + t_string + " -------")

            self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                orders=self.orders,
                positions=self.positions,
                data=price,
                time=t,
                wallet=self.wallet
            )

            # ----------------- Print the results ----------------- #
            if env == "local":
                print(f"Wallet: {self.wallet}")
                print(f"Price: {price}")
                print(f"Orders[0]: {self.orders[0]}")

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

            # Toral value of positions
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

            iter_log = 0
            if t.second == 30 and t.minute % 2 == 0 and not notif_sent:
                title = emoji.emojize(":clock1:") + " Recurrent update " + emoji.emojize(":clock1:") + "\n\n"
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
                    # Record pnl and wallet each 10 minutes
                    # Check if file exists
                    if not os.path.exists("io/live_testing/perf/" + self.strategy.__str__() + ".csv"):
                        subprocess.run(["touch", "io/live_testing/perf/" + self.strategy.__str__() + ".csv"])
                        with open("io/live_testing/perf/" + self.strategy.__str__() + ".csv", "w") as f:
                            f.write("iter,time,open_pos,closed_pos,tot_pos,"
                                    "pnl_pos_closed,pnl_pos_open,pnl_pos_tot,wallet_tot\n")

                    # Write the data
                    with open("io/live_testing/perf/" + self.strategy.__str__() + ".csv", "a") as f:
                        f.write(f"{iter_log},"
                                f"{t},"
                                f"{open_pos_},"
                                f"{len(self.positions) - open_pos_},"
                                f"{len(self.positions)},"
                                f"{pnl_closed},"
                                f"{pnl_open},"
                                f"{tot_ + self.wallet}\n")

            if env == "local":
                time.sleep(5)
            else:
                time.sleep(0.25)
