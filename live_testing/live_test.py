import asyncio
import websockets
import requests
import time
import json

from utils import *

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
                print(index_price)
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
        return price

    def run(self):
        # Every 5 seconds get the index price
        while True:
            price = self.get_price()
            self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                price, self.orders, self.positions, self.wallet
            )
            print(f"Wallet: {self.wallet}")
            print(f"Orders[0]: {self.orders[0]}")
            print(f"Positions: {self.positions}")
            time.sleep(5)
