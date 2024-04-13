import asyncio
import websockets
import requests
import time
import json

from utils.utils import *

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
        return price, time.time()

    def run(self):
        # Every 5 seconds get the index price
        while True:

            price, t = self.get_price()

            print_yellow(f"------- Update at {t} -------")

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

            time.sleep(5)
