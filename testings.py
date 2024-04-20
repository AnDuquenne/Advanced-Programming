#%%
import asyncio
import websockets
import json
import datetime
import requests

import os, sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import *

def get_historical_price(days: int, minutes: int, interval: str = "5"):
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
    :param interval: interval of the candles
    :return:
    """
    # Get the current timestamp and the timestamp of the last x days
    t0, t1 = get_timestamps(datetime.timedelta(days=days, minutes=minutes))

    url = "https://api-testnet.bybit.com/v5/market/mark-price-kline"

    # Index does not have an underscore on bybit, and are written in uppercase
    bybit_index = "eth_usdt".upper().replace("_", "")

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

if __name__ == "__main__":
    price = get_historical_price(0, 30)
    print(price)