import numpy as np


class EMA:
    """
    A class defining the Exponential Moving Average (EMA) of a time series.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.ema = np.zeros(len(self.time_series))

        self.update_ema()

    def EMA_t(self, t):
        """
        Calculate the EMA at time t.
        """
        if t == 0:
            return 0
        else:
            return (self.time_series[t] - self.ema[t - 1]) * (2 / (self.window + 1)) + self.ema[t - 1]

    def update_ema(self):
        """
        Update the EMA.
        """
        for t in range(1, len(self.time_series)):
            self.ema[t] = self.EMA_t(t)

    @property
    def get_ema(self):
        """
        Return the EMA values.
        """
        return self.ema


class WMA:
    """
    A class defining the Weighted Moving Average (WMA) of a time series.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.wma = np.zeros(len(self.time_series))

        self.update_wma()

    def update_wma(self):
        """
        Update the WMA.
        """
        for t in range(self.window, len(self.time_series)):
            weights = np.arange(1, self.window + 1)
            self.wma[t] = np.dot(self.time_series[t - self.window:t], weights) / sum(weights)

    @property
    def get_wma(self):
        """
        Return the WMA values.
        """
        return np.round(self.wma, 4)


class MACD:
    """
    A class defining the Moving Average Convergence Divergence (MACD) of a time series.
    """

    def __init__(self, time_series):
        self.time_series = time_series
        self.ema_12 = EMA(self.time_series, 12)
        self.ema_26 = EMA(self.time_series, 26)
        self.macd = self.ema_12.get_ema - self.ema_26.get_ema
        self.signal_line = EMA(self.macd, 9).get_ema
        self.histogram = self.macd - self.signal_line


    def get_macd(self):
        """
        Return the MACD values.
        """
        return np.round(self.macd, 4)


    def get_signal_line(self):
        """
        Return the signal line values.
        """
        return np.round(self.signal_line, 4)


    def get_histogram(self):
        """
        Return the histogram values.
        """
        return np.round(self.histogram, 4)


class RSI:
    """
    A class defining the Relative Strength Index (RSI) of a time series.

    1. Calculate the average gain and average loss over the last n periods.
    2. Calculate the relative strength (RS).
    3. Calculate the RSI.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.rsi = np.zeros(len(self.time_series))

        self.diff = np.diff(self.time_series)

        self.update_rsi()

    def average_gain(self, t):
        """
        Calculate the average gain at time t.
        """
        # Using vectorized operations

        window_period = self.diff[t - self.window:t]
        average_gains = np.sum(window_period[window_period > 0] / self.window)
        average_losses = -np.sum(window_period[window_period < 0] / self.window)

        return average_losses, average_gains

    def RS(self, t):
        """
        Calculate the relative strength at time t.
        """
        average_loss, average_gain = self.average_gain(t)

        return average_gain / average_loss

    def update_rsi(self):
        """
        Update the RSI.
        """
        for t in range(1, len(self.time_series)):
            RS = self.RS(t)

            self.rsi[t] = 100 - (100 / (1 + RS))
            # if RS == 0 or RS == np.inf or RS == np.nan or RS == -np.inf:
            #     print(t, RS, self.rsi[t])


    def get_rsi(self):
        """
        Return the RSI values.
        """
        return np.round(self.rsi, 4)


class StochasticRSI:
    """
    A class defining the Stochastic RSI of a time series.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.stochastic_rsi = np.zeros(len(self.time_series))

        self.RSI = RSI(self.time_series, self.window).get_rsi

        self.update_stochastic_rsi()

    def update_stochastic_rsi(self):
        """
        Update the Stochastic RSI.
        """
        for t in range(self.window, len(self.time_series)):
            min_RSI = min(self.RSI[t - self.window:t])
            max_RSI = max(self.RSI[t - self.window:t])
            self.stochastic_rsi[t] = (self.RSI[t] - min_RSI) / (max_RSI - min_RSI) if max_RSI - min_RSI != 0 else 0


    def get_stochastic_rsi(self):
        """
        Return the Stochastic RSI values.
        """
        return np.round(self.stochastic_rsi, 4)


class DPO:
    """
    A class defining the Detrended Price Oscillator (DPO) of a time series.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.dpo = np.zeros(len(self.time_series))

        self.update_dpo()

    def update_dpo(self):
        """
        Update the DPO.
        """
        for t in range(self.window, len(self.time_series)):
            simple_moving_average = np.mean(self.time_series[t - self.window:t])
            self.dpo[t] = self.time_series[t - self.window // 2] - simple_moving_average


    def get_dpo(self):
        """
        Return the DPO values.
        """
        return np.round(self.dpo, 4)


class CC:
    """
    A class defining the Coppock Curve (CC) of a time series.
    """

    def __init__(self, time_series, window):
        self.time_series = time_series
        self.window = window
        self.cc = np.zeros(len(self.time_series))

        self.update_cc()

    def update_cc(self):
        """
        Update the CC. Use the 14 and 11 periods ROC.
        """
        for t in range(self.window, len(self.time_series)):
            roc_14 = (self.time_series[t] - self.time_series[t - 14]) / self.time_series[t - 14]
            roc_11 = (self.time_series[t] - self.time_series[t - 11]) / self.time_series[t - 11]
            self.cc[t] = roc_14 + roc_11

        # 10 periods weighted moving average
        self.cc = WMA(self.cc, 10).get_wma


    def get_cc(self):
        """
        Return the CC values.
        """
        return np.round(self.cc, 4)
