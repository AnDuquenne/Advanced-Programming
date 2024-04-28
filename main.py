from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI

import pandas as pd
import os
from tqdm import tqdm
from utils.data_handler import DataCleaner
if __name__ == "__main__":

    # create a list of files that contain 2018 or 2019 or 2020 or 2021 or 2022 in their name
    files = [f for f in os.listdir("Data/crypto-1min/eth-min1/cleaned_data")]

    for file in files:
        print("'Data/crypto-1min/eth-min1/cleaned_data/" + file + "',")
