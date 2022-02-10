from util import get_data, plot_data
import pandas as pd
import datetime as dt

def testPolicy(
    symbol="JPM",
    sd = dt.datetime(2008, 1, 1),
    ed = dt.datetime(2009, 12, 31),
    sv =100000,
    commission=0.0,
    impact=0.00
):
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices[symbol]
    differences = prices[1:].values - prices[:-1].values
    differences = [2000 if x > 0 else -2000 for x in differences]

    prev = differences[0]
    trades = []
    trades.append(prev)
    for x in differences[1:]:
        if x == prev:
            trades.append(0)
        else:
            trades.append(x)
            prev = x

    trades.append(-1*prev)
    trades[0] = trades[0] * .5

    orders = pd.DataFrame(index=prices.index)
    orders['Shares'] = trades

    return orders

def author():
    return "jchen384"

if __name__ == "__main__":
    testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)

