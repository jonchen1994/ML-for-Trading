import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
import marketsimcode as mktsim
import StrategyLearner as sl
import pandas as pd
import experiment1 as exp1
import experiment2 as exp2
import ManualStrategy as ms
import marketsimcode as mktsim

def report_info():
    #In and Out Samples
    in_sd = dt.datetime(2008, 1, 1)
    in_ed = dt.datetime(2009, 12, 31)
    out_sd = dt.datetime(2010, 1, 1)
    out_ed = dt.datetime(2011, 12, 31)
    symbol = 'JPM'
    sv = 100000
    commission = 9.95
    impact = .005

    Man_Sum_Stats = pd.DataFrame(columns=['Sample Type', 'Cumulative Ret', 'STD', 'Mean'])

    #Manual Strategy for In Sample
    df_manual = ms.testPolicy(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
    df_manual['Symbol'] = symbol
    manPort = mktsim.compute_portvals(df_manual, start_val=sv, commission=commission, impact=impact)

    man_buy = [i for i, x in df_manual.iterrows() if x['Shares'] > 0]
    man_sell = [i for i, x in df_manual.iterrows() if x['Shares'] < 0]

    # Benchmark
    benchmark = df_manual.copy()
    benchmark['Shares'] = 0
    benchmark['Shares'].iloc[0] = 1000
    benchPort = mktsim.compute_portvals(benchmark, start_val=sv, commission=commission, impact=impact)

    plot_port = df_manual.copy()
    plot_port['Benchmark'] = benchPort.values / benchPort.values[0]
    plot_port['Manual'] = manPort.values / manPort.values[0]
    plot_port = plot_port[['Benchmark', 'Manual']]

    manCRet = (plot_port['Manual'].iloc[-1]-plot_port['Manual'].iloc[0])/plot_port['Manual'].iloc[0]
    benchCRet = (plot_port['Benchmark'].iloc[-1]-plot_port['Benchmark'].iloc[0])/plot_port['Benchmark'].iloc[0]
    manSTD = plot_port['Manual'].std()
    manavg = plot_port['Manual'].mean()
    benchSTD = plot_port['Benchmark'].std()
    benchavg = plot_port['Benchmark'].mean()

    Man_Sum_Stats.loc[0]= ['Manual In Sample', manCRet, manSTD, manavg]
    Man_Sum_Stats.loc[1] = ['Benchmark In Sample', benchCRet, benchSTD, benchavg]


    plt.figure(figsize=(14, 8))
    plt.plot(plot_port['Benchmark'], color='green')
    plt.plot(plot_port['Manual'], color='red')

    for xc in man_buy:
        plt.axvline(x=xc, color = 'blue')
    for xc in man_sell:
        plt.axvline(x=xc, color='black')

    plt.xlim([in_sd, in_ed])
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.title("Manual vs Benchmark - For In Sample")
    plt.legend(plot_port.columns, loc='upper left')
    plt.savefig('ManualInSample.png')

    # Manual Strategy for Out Sample
    df_manual = ms.testPolicy(symbol=symbol, sd=out_sd, ed=out_ed, sv=sv)
    df_manual['Symbol'] = symbol
    manPort = mktsim.compute_portvals(df_manual, start_val=sv, commission=commission, impact=impact)

    man_buy = [i for i, x in df_manual.iterrows() if x['Shares'] > 0]
    man_sell = [i for i, x in df_manual.iterrows() if x['Shares'] < 0]

    # Benchmark
    benchmark = df_manual.copy()
    benchmark['Shares'] = 0
    benchmark['Shares'].iloc[0] = 1000
    benchPort = mktsim.compute_portvals(benchmark, start_val=sv, commission=commission, impact=impact)

    plot_port = df_manual.copy()
    plot_port['Benchmark'] = benchPort.values / benchPort.values[0]
    plot_port['Manual'] = manPort.values / manPort.values[0]
    plot_port = plot_port[['Benchmark', 'Manual']]

    manCRet = (plot_port['Manual'].iloc[-1] - plot_port['Manual'].iloc[0]) / plot_port['Manual'].iloc[0]
    benchCRet = (plot_port['Benchmark'].iloc[-1] - plot_port['Benchmark'].iloc[0]) / plot_port['Benchmark'].iloc[0]
    manSTD = plot_port['Manual'].std()
    manavg = plot_port['Manual'].mean()
    benchSTD = plot_port['Benchmark'].std()
    benchavg = plot_port['Benchmark'].mean()

    Man_Sum_Stats.loc[2] = ['Manual Out Sample', manCRet, manSTD, manavg]
    Man_Sum_Stats.loc[3] = ['Benchmark Out Sample', benchCRet, benchSTD, benchavg]

    plt.figure(figsize=(14, 8))
    plt.plot(plot_port['Benchmark'], color='green')
    plt.plot(plot_port['Manual'], color='red')

    for xc in man_buy:
        plt.axvline(x=xc, color='blue')
    for xc in man_sell:
        plt.axvline(x=xc, color='black')

    plt.xlim([out_sd, out_ed])
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.title("Manual vs Benchmark - For Out of Sample")
    plt.legend(plot_port.columns, loc='upper left')
    plt.savefig('ManualOutSample.png')



    exp1.experiment1(in_sd, in_ed, out_sd, out_ed, symbol, sv, commission, impact)

    commission = 0
    impacts = [0, 0.0025, 0.005, 0.01]

    exp2_Stats = exp2.experiment2(in_sd, in_ed, symbol, sv, commission, impacts)

    return Man_Sum_Stats, exp2_Stats

if __name__ == "__main__":
    Man_Sum_Stats, exp2_Stats = report_info()
    print(Man_Sum_Stats)
    print(exp2_Stats)

def author():
    return 'jchen384'