
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import ManualStrategy as ms
import marketsimcode as mktsim
import StrategyLearner as sl

def experiment1(in_sd, in_ed, out_sd, out_ed, symbol, sv, commission, impact):
    #Setting up

    prices = get_data([symbol], pd.date_range(in_sd, in_ed))
    prices = prices[symbol] / prices[symbol].iloc[0]

    #Manual Strategy
    df_manual = ms.testPolicy(symbol=symbol, sd = in_sd, ed= in_ed, sv = sv)
    df_manual['Symbol'] = symbol
    manPort = mktsim.compute_portvals(df_manual, start_val=sv, commission=commission, impact=impact)

    #ML Strategy
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    learner.add_evidence(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
    df_strat = learner.testPolicy(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
    df_strat['Symbol'] = symbol
    stratPort = mktsim.compute_portvals(df_strat, start_val=sv, commission=commission, impact=impact)

    #Benchmark
    benchmark = df_manual.copy()
    benchmark['Shares'] = 0
    benchmark['Shares'].iloc[0] = 1000
    benchPort = mktsim.compute_portvals(benchmark, start_val=sv, commission=commission, impact=impact)

    #Normalize and plot
    plot_port = df_manual.copy()
    plot_port['Benchmark'] = benchPort.values / benchPort.values[0]
    plot_port['Manual'] = manPort.values / manPort.values[0]
    plot_port['Strategy'] = stratPort.values / stratPort.values[0]
    plot_port = plot_port[['Benchmark', 'Manual','Strategy']]

    plt.figure(figsize=(14, 8))
    plt.plot(plot_port['Benchmark'], color='red')
    plt.plot(plot_port['Manual'], color='blue')
    plt.plot(plot_port['Strategy'], color='green')
    plt.xlim([in_sd, in_ed])
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.title("Benchmark vs Manual vs Strategy")
    plt.legend(plot_port.columns, loc='upper left')
    plt.savefig('experiment1.png')

if __name__ == "__main__":
    in_sd = dt.datetime(2008, 1, 1)
    in_ed = dt.datetime(2009, 12, 31)
    out_sd = dt.datetime(2010, 1, 1)
    out_ed = dt.datetime(2011, 12, 31)
    symbol = 'JPM'
    sv = 100000
    commission = 9.95
    impact = .005

    experiment1(in_sd, in_ed, out_sd, out_ed, symbol, sv, commission, impact)

def author():
    return 'jchen384'


# #Boll #1

    # boll_sig = get_bollinger("JPM", sd, ed)
    # boll_votes = pd.DataFrame(index=boll_sig.index)
    # boll_votes['votes'] = boll_vote(boll_sig)

    # boll_buy = [i for i, x in boll_votes.iterrows() if x['votes'] == 2]
    # boll_sell = [i for i, x in boll_votes.iterrows() if x['votes'] == -2]
    # boll_neut = [i for i, x in boll_votes.iterrows() if abs(x['votes']) == 1]

    # mom_sig = get_momentum("JPM", sd, ed, window = 20)
    # mom_plot = mom_sig -.5
    # mom_votes = pd.DataFrame(index=mom_sig.index)
    # mom_votes['states'],mom_votes['actions'] = mom_vote(mom_sig)

    # mom_buy = [i for i, x in mom_votes.iterrows() if x['actions'] == 2]
    # mom_sell = [i for i, x in mom_votes.iterrows() if x['actions'] == -2]
    # mom_neut = [i for i, x in mom_votes.iterrows() if abs(x['actions']) == 1]

    # mac_sig = get_MACD("JPM", sd, ed, 26, 12, 9)
    # diff = mac_sig['Difference']/(3* max(mac_sig['Difference'])) - 1.5
    # signal = mac_sig['Signal'] / (3*max(mac_sig['Signal'])) - 1.5
    # mac_votes = pd.DataFrame(index=mac_sig.index)
    # mac_votes['actions'] = mac_vote(mac_sig)
    # mac_buy = [i for i, x in mac_votes.iterrows() if x['actions'] == 2]
    # mac_sell = [i for i, x in mac_votes.iterrows() if x['actions'] == -2]

    # full_strat = pd.DataFrame(index=mac_sig.index)
    # full_strat['Actions'] = strategy(boll_votes, mom_votes, mac_votes)
    # strat_buy = [i for i, x in full_strat.iterrows() if x['actions'] == 1]
    # strat_sell = [i for i, x in full_strat.iterrows() if x['actions'] == -1]




    #visualization
    # bol = bollingerBand(prices)
    # stds = prices.rolling(20).std()
    # ma = SMA(prices, 20)
    # upBol = ma+(2*stds)
    # lowBol = ma-(2*stds)
    # plt.figure(figsize=(14, 8))
    # plt.plot(prices, color='black')
    # plt.plot(upBol, color='blue')
    # plt.plot(lowBol, color='blue')

    #BOLL trading
    # for xc in boll_buy:
    #     plt.axvline(x=xc, color = 'green')
    # for xc in boll_sell:
    #     plt.axvline(x=xc, color='red')
    # for xc in boll_neut:
    #     plt.axvline(x=xc, color='orange')

    #Momentum trading
    # for xc in mom_buy:
    #     plt.axvline(x=xc, color = 'green')
    # for xc in mom_sell:
    #     plt.axvline(x=xc, color='red')
    # for xc in mom_neut:
    #     plt.axvline(x=xc, color='orange')

    # plt.plot(mom_plot, color='purple')
    # plt.axhline(-.5, color='purple', linestyle='--')

    #MACD trading
    # for xc in mac_buy:
    #     plt.axvline(x=xc, color = 'green')
    # for xc in mac_sell:
    #     plt.axvline(x=xc, color='red')
    # for xc in mom_neut:
    #     plt.axvline(x=xc, color='orange')

    # plt.plot(diff, color='brown')
    # plt.plot(signal, color='gray')

    #Strat trading
    # for xc in strat_buy:
    #     plt.axvline(x=xc, color = 'green')
    # for xc in strat_sell:
    #     plt.axvline(x=xc, color='red')


    # plt.xlim([sd, ed])
    # plt.xlabel("Dates")
    # plt.ylabel("Normalized Prices")
    # plt.title("Prices vs. Bollinger Bands")
    # plt.legend(['Prices','Bollinger Bands'], loc='upper left')
    # plt.show()