from util import get_data, plot_data
import datetime as dt
import os
import numpy as np
import pandas as pd

def compute_portvals(
        order,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):

    orders = order.copy()
    orders['Date'] = pd.to_datetime(orders.index)

    start_date = orders['Date'].min()
    end_date = orders['Date'].max()

    symbols = orders['Symbol'].unique()

    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices = prices[symbols]  # remove SPY
    prices['CASH'] = 1

    # Trades DF
    trades = make_trades(prices, orders, symbols, commission, impact)

    # Holding DF
    holdings = make_holdings(trades, start_val, symbols)

    # value DF
    values = get_values(prices, holdings)

    # Portfolio
    portval = values.sum(axis=1)

    return portval


def make_holdings(trades, start_val, symbols):
    holdings = trades.copy()
    holdings[symbols] = 0
    holdings['CASH'] = 0

    prev_row = np.zeros(len(symbols) + 1)
    prev_row[-1] += start_val

    for index, row in trades.iterrows():
        curr_row = row + prev_row
        holdings.loc[index] += curr_row
        prev_row = curr_row

    return holdings


def get_values(prices, holdings):
    return prices * holdings


def make_trades(prices, orders, symbols, commission, impact):
    trades = prices.copy()
    trades[symbols] = 0.0
    trades['CASH'] = 0.0

    for index, row in orders.iterrows():
        temp_date = row['Date']
        temp_sym = row['Symbol']
        temp_shares = row['Shares']
        temp_order = 'BUY' if temp_shares >=0 else 'SELL'
        temp_prices = prices.loc[temp_date]
        temp_trade = trades.loc[temp_date]
        if temp_order == 'BUY':
            coef = 1
        else:
            coef = -1

        value = temp_shares * temp_prices[temp_sym]
        temp_trade['CASH'] += (-1 * value) - (impact * abs(value)) - commission
        temp_trade[temp_sym] += temp_shares

    return trades


def author():
    return "jchen384"


if __name__ == "__main__":
    print('test')
