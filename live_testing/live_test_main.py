from live_test import LiveTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_MACD import StrategyMACD

import argparse

import datetime


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

    # Create the live test object
    live_test = LiveTest("eth_usd", strategy, wallet)

    # Run the live test
    live_test.run()
