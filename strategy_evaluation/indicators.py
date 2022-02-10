from util import get_data, plot_data
import pandas as pd
import numpy as np


def bollingerBand(prices, window = 20, width = 2):
    stds = prices.rolling(window).std()
    ma = SMA(prices,window)
    return (prices - ma) / (stds * width)

def SMA(prices, window):
    prices = prices/prices.iloc[0]
    return prices.rolling(window).mean()

def momentum(prices, window):
    data = prices.copy()
    data.iloc[window:] = data.iloc[window:]/data.iloc[:-window].values - 1
    data.iloc[:window] = np.nan
    return data

def EMA(prices, span):
    return prices.ewm(span = span).mean()

def MACD(prices, quick = 26, slow = 12, signal = 9):
    slow_ema = prices.ewm(span=slow).mean()
    quick_ema = prices.ewm(span=quick).mean()
    difference = slow_ema - quick_ema
    smoothed_ema = difference.ewm(span = signal).mean()
    signal = pd.concat([difference, smoothed_ema], axis=1)
    signal.columns = ['Difference','Signal']
    return signal

def author():
    return "jchen384"

if __name__ == "__main__":
    print('test')
