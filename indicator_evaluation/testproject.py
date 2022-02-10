# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from util import get_data, plot_data
from indicators import MACD, EMA, momentum, SMA, bollingerBand
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import TheoreticallyOptimalStrategy as tos
import marketsimcode as mktsim

def author():
    return 'jchen384'

def dailyRet(prices):
    daily = prices.copy()
    daily[1:] = (prices[1:]/prices[:-1].values) - 1
    daily.iloc[0] = 0
    return daily

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #TOS
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades['Symbol'] = "JPM"
    print(df_trades)
    optPort = mktsim.compute_portvals(df_trades, start_val=100000, commission=0, impact= 0)
    benchmark = df_trades.copy()
    benchmark['Shares'] = 0
    benchmark['Shares'].iloc[0] = 1000
    benchPort = mktsim.compute_portvals(benchmark, start_val=100000, commission=0, impact= 0)
    #
    plot_port = df_trades.copy()
    plot_port['Benchmark'] = benchPort.values/benchPort.values[0]
    plot_port['Optimal'] = optPort.values/optPort.values[0]
    plot_port = plot_port[['Benchmark','Optimal']]

    plt.figure(figsize=(14, 8))
    plt.plot(plot_port['Benchmark'], color = 'green')
    plt.plot(plot_port['Optimal'], color='red')
    plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.title("Optimal vs Benchmark")
    plt.legend(plot_port.columns, loc='upper left')
    plt.savefig('OptvsBench.png')


    opt_cumul = optPort[-1] / optPort[0] - 1
    bench_cumul = benchPort[-1] / benchPort[0] - 1
    print("optimal cumulative return is " + str(opt_cumul))
    print("benchmark cumulative return is " + str(bench_cumul))
    opt_daily = dailyRet(optPort)
    bench_daily = dailyRet(benchPort)

    print("optimal std is "+ str(opt_daily.std()))
    print("benchmark std is " + str(bench_daily.std()))
    print("optimal mean is " + str(opt_daily.mean()))
    print("bench mean is " + str(bench_daily.mean()))

#####END TOS##########
    # prices = get_data(['JPM'], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))
    # prices = prices['JPM']/prices['JPM'].iloc[0]
    #
    # #SMA
    # movAvg = SMA(prices, 20)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='red')
    # plt.plot(movAvg, color='green')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. SMA")
    # plt.legend(['Prices','SMA'], loc='upper left')
    # plt.savefig('SMA.png')
    #
    # #Momentum
    # mom = momentum(prices, 5)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='red')
    # plt.plot(mom, color='green')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices and Momentum")
    # plt.title("Prices vs. Momentum")
    # plt.legend(['Prices','Momentum'], loc='upper left')
    # plt.savefig('Momentum.png')
    #
    # #EMA
    # expAvg = EMA(prices, 20)
    # movAvg = SMA(prices, 20)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='red')
    # plt.plot(expAvg, color='green')
    # plt.plot(movAvg, color='orange')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. EMA")
    # plt.legend(['Prices','EMA','SMA'], loc='upper left')
    # plt.savefig('EMA.png')
    #
    # #MACD
    # bigMac = MACD(prices)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='red')
    # plt.plot(bigMac['Difference'], color='green')
    # plt.plot(bigMac['Signal'], color='orange')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. MACD")
    # plt.legend(['Prices','MACD','Signal'], loc='upper left')
    # plt.savefig('MACD.png')
    # #
    # # #Boll #1
    # bol = bollingerBand(prices)
    # stds = prices.rolling(20).std()
    # ma = SMA(prices, 20)
    # upBol = ma+(2*stds)
    # lowBol = ma-(2*stds)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='red')
    # plt.plot(upBol, color='blue')
    # plt.plot(lowBol, color='blue')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. Bollinger Bands")
    # plt.legend(['Prices','Bollinger Bands'], loc='upper left')
    # plt.savefig('Boll1.png')
    # #
    # # #Boll #2
    # plt.figure(figsize=(14, 8))
    # bolnorm = bol/bol[19] + 1
    # plt.plot(prices, color='red')
    # plt.plot(bolnorm, color='green')
    # plt.xlim([dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. Bollinger Bands Signal ")
    # plt.legend(['Prices','Bollinger Bands Signal'], loc='upper left')
    # plt.savefig('Boll2.png')

