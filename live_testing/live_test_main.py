from live_test import LiveTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_MACD import StrategyMACD
from Strategies.Strategy_FC import StrategyFC

import argparse

import datetime
import yaml

# Load the configuration file
with open("../io/config.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Live test')
    parser.add_argument('--strategy', type=str, help='The strategy to use')
    parser.add_argument('--buy_percentage', type=float, default=0.01, help='Buy percentage')
    parser.add_argument('--wallet', type=float, default=1000, help='The wallet amount')
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

    # Create the live test object
    live_test = LiveTest("eth_usd", strategy, wallet)

    # Run the live test
    live_test.run()
