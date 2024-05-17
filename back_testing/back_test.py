import time
import datetime
import json

import emoji
from tqdm import tqdm
import csv

import sys
import os
import subprocess

# load environment variables
from dotenv import load_dotenv

load_dotenv()
env = os.getenv("environment")

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import *
from utils.tech_analysis import MACD
from utils.notifications import send_message

from Strategies.SFStrategy import SFStrategy
from Strategies.SFStrategy_I import SFStrategyI
from Strategies.Strategy_FC import StrategyFC
from Strategies.Strategy_LSTM import StrategyLSTM
from Strategies.Strategy_MACD import StrategyMACD


class BackTest:
    def __init__(self, data, strategy, wallet):
        self.df = data

        self.orders = []
        self.positions = []
        self.wallet = wallet

        self.strategy = strategy

    def run(self):
        """
        Used to perform backtest on the data recovered from the cedif and kaggle.
        :return:
        """
        time_sequence_length = self.df['close'].shape[0]

        results = []

        # For each time step check if orders should be executed or if positions should be closed
        # Strating at 10 to avoid out of range error with networks
        for i in tqdm(range(10, time_sequence_length)):

            if isinstance(self.strategy, StrategyFC):
                self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                    self.orders, self.positions, self.df['close'][i-10:i], self.df['date'][i], self.wallet, 1)
            else:
                self.orders, self.positions, self.wallet = self.strategy.check_conditions(
                    self.orders, self.positions, self.df['close'][i], self.df['date'][i], self.wallet
                )

            # Open positions
            open_pos_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    open_pos_ += 1

            # PNLs
            # PNL of closed pos
            pnl_closed = 0
            pnl_open = 0
            for pos_ in self.positions:
                if pos_.status == 'closed':
                    pnl_closed += pos_.profit
                elif pos_.status == 'open':
                    pnl_open += pos_.pnl(self.df['close'][i])

            # Total value of positions
            tot_ = 0
            for pos_ in self.positions:
                if pos_.status == 'open':
                    tot_ += pos_.dollars_value(self.df['close'][i])

            # Save the results in the dict
            results.append({
                'time': self.df['date'][i],
                'wallet': self.wallet,
                'open_pos': open_pos_,
                'closed_pos': len(self.positions) - open_pos_,
                'total_pos': len(self.positions),
                'pnl_closed': pnl_closed,
                'pnl_open': pnl_open,
                'total_pnl': pnl_closed + pnl_open,
                'total_value': tot_ + self.wallet
            })

        # Save the results in a csv file
        fields = results[0].keys()
        if isinstance(self.strategy, SFStrategyI):
            with open('../io/back_test/strategies_time_evolution/SFStrategyI/' + self.strategy.__str__() + '.csv', 'w', newline="") as f:
                write = csv.DictWriter(f, fieldnames=fields)
                write.writeheader()
                write.writerows(results)
        elif isinstance(self.strategy, SFStrategy):
            with open('../io/back_test/strategies_time_evolution/SFStrategy/' + self.strategy.__str__() + '.csv', 'w', newline="") as f:
                write = csv.DictWriter(f, fieldnames=fields)
                write.writeheader()
                write.writerows(results)
        elif isinstance(self.strategy, StrategyFC):
            with open('../io/back_test/strategies_time_evolution/StrategyFC/' + self.strategy.__str__() + '.csv', 'w', newline="") as f:
                write = csv.DictWriter(f, fieldnames=fields)
                write.writeheader()
                write.writerows(results)