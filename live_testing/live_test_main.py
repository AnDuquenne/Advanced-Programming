from live_test import LiveTest
from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI

strategy = SFStrategyI(buy_percentage=0.00025)
wallet = 1000

if __name__ == "__main__":
    # Create the live test object
    live_test = LiveTest("eth_usd", strategy, wallet)

    # Run the live test
    live_test.run()
