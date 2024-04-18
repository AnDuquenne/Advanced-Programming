from live_test import LiveTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI

import datetime

run_name = "ETH_USD_" + datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

strategy = SFStrategyI(run_name=run_name, buy_percentage=0.00025)
wallet = 1000

if __name__ == "__main__":
    # Create the live test object
    live_test = LiveTest("eth_usd", strategy, wallet)

    # Run the live test
    live_test.run()
