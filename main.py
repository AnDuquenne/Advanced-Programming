from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI

if __name__ == "__main__":
    files = ["Data/crypto-1min/btc-min1/BTC_MACD_december2023.csv",
             "Data/crypto-1min/btc-min1/BTC_MACD_January2024.csv",
             "Data/crypto-1min/btc-min1/BTC_MACD_february2024.csv",
             "Data/crypto-1min/btc-min1/BTC_MACD_march2024.csv",
             "Data/crypto-1min/eth-min1/ETH_MACD_december2023.csv",
             "Data/crypto-1min/eth-min1/ETH_MACD_January2024.csv",
             "Data/crypto-1min/eth-min1/ETH_MACD_february2024.csv",
             "Data/crypto-1min/eth-min1/ETH_MACD_march2024.csv"
             ]

    # for file in files:
    #     s = SFStrategy(file)
    #     s.run()
    #
    #     s = SFStrategyI(file)
    #     s.run()

    s = SFStrategy(files[5])
    s.run()
    s.show_simulation()
