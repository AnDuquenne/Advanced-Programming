from back_test import BackTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_MACD import StrategyMACD

import argparse
import pandas as pd

import datetime
import yaml

import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Load the configuration file
with open("../io/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


if __name__ == "__main__":

    strat_ = cfg["back_test"]["main"]["strategy"]
    buy_percentage_ = cfg["back_test"]["main"]["buy_percentage"]
    wallet = cfg["back_test"]["main"]["wallet"]

    data = pd.read_csv("../" + cfg["back_test"]["main"]["data_path"])

    # Transform the date column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')

    run_name = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + strat_ + "__" + str(buy_percentage_)

    strategy = None
    if strat_ == "SFStrategyI":
        strategy = SFStrategyI(run_name=run_name, buy_percentage=buy_percentage_)
    elif strat_ == "SFStrategy":
        strategy = SFStrategy(run_name=run_name, buy_percentage=buy_percentage_)
    elif strat_ == "StrategyMACD":
        strategy = StrategyMACD(run_name=run_name, buy_percentage=buy_percentage_)

    # Create the live test object
    back_test = BackTest(data, strategy, wallet)

    # Run the live test
    back_test.run()
