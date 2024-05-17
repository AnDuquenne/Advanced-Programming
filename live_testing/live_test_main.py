from live_test import LiveTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_MACD import StrategyMACD
from Strategies.Strategy_FC import StrategyFC
from Strategies.Strategy_LSTM import StrategyLSTM

import argparse

import datetime
import yaml

# load environment variables
from dotenv import load_dotenv

import sys
import os

load_dotenv()
env = os.getenv("environment")

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

    parser = argparse.ArgumentParser(description='Live test')
    parser.add_argument('--strategy', type=str, help='The strategy to use')
    parser.add_argument('--buy_percentage', type=float, default=0.01, help='Buy percentage')
    parser.add_argument('--wallet', type=float, default=1000.00, help='The wallet amount')
    args = parser.parse_args()

    strat_ = args.strategy
    buy_percentage_ = args.buy_percentage
    wallet = args.wallet

    run_name = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + strat_ + "__" + str(buy_percentage_)

    strategy = None
    if strat_ == "SFStrategyI":
        strategy = SFStrategyI(run_name=run_name, buy_percentage=buy_percentage_)
    elif strat_ == "SFStrategy":
        strategy = SFStrategy(run_name=run_name, buy_percentage=buy_percentage_)
    elif strat_ == "StrategyMACD":
        strategy = StrategyMACD(run_name=run_name, buy_percentage=buy_percentage_)
    elif strat_ == "StrategyFC":
        network_params = {
            "input_size": cfg["live_test"]["main"]["fc"]["window"],
            "hidden_size": cfg["live_test"]["main"]["fc"]["hidden_size"],
            "dropout": cfg["live_test"]["main"]["fc"]["dropout"],
            "device": cfg["live_test"]["main"]["fc"]["device"],
            "weights_path": cfg["live_test"]["main"]["fc"]["weights_path"],
        }
        strategy = StrategyFC(run_name=run_name, network_params=network_params, buy_percentage=buy_percentage_,
                              debug=False)
    elif strat_ == "StrategyLSTM":
        network_params = {
            "input_size": cfg["live_test"]["main"]["lstm"]["window"],
            "hidden_size": cfg["live_test"]["main"]["lstm"]["hidden_size"],
            "num_layers": cfg["live_test"]["main"]["lstm"]["num_layers"],
            "dropout": cfg["live_test"]["main"]["lstm"]["dropout"],
            "device": cfg["live_test"]["main"]["lstm"]["device"],
            "weights_path": cfg["live_test"]["main"]["lstm"]["weights_path"],
        }
        strategy = StrategyLSTM(run_name=run_name, network_params=network_params, buy_percentage=buy_percentage_,
                                debug=False)

    # Create the live test object
    live_test = LiveTest("btc_usd", strategy, wallet)

    # Run the live test
    live_test.run()
