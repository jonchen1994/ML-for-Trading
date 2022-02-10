import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
import marketsimcode as mktsim
import StrategyLearner as sl
import pandas as pd

def experiment2(in_sd, in_ed, symbol, sv, commission, impacts):

    df = pd.DataFrame(columns=['Impact','Num_Zeroes','STD','Mean'])

    dumb = True
    for i in range(len(impacts)):
        impact = impacts[i]
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
        learner.add_evidence(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
        df_strat = learner.testPolicy(symbol=symbol, sd=in_sd, ed=in_ed, sv=sv)
        df_strat['Symbol'] = symbol
        stratPort = mktsim.compute_portvals(df_strat, start_val=sv, commission=commission, impact=impact)
        num_z = len(df_strat[df_strat['Shares'] == 0])
        stand_dev = stratPort.std()
        average = stratPort.mean()
        df.loc[i] = [impact, num_z, stand_dev, average]

        if dumb:
            plot_port = df_strat.copy()
            dumb = False

        plot_port[str(i)] = stratPort.values/stratPort.values[0]

    print(plot_port)
    plot_port = plot_port[['0','1','2','3']]
    plot_port.columns = ['Impact:0','Impact:0.0025','Impact:0.005','Impact:.01']

    plt.figure(figsize=(14, 8))
    plt.plot(plot_port['Impact:0'], color='red')
    plt.plot(plot_port['Impact:0.0025'], color='blue')
    plt.plot(plot_port['Impact:0.005'], color='green')
    plt.plot(plot_port['Impact:.01'], color='orange')
    plt.xlim([in_sd, in_ed])
    plt.xlabel("Dates")
    plt.ylabel("Returns")
    plt.title("Impact Comparison")
    plt.legend(plot_port.columns, loc='upper left')
    plt.savefig('experiment2.png')
    return df

if __name__ == "__main__":
    in_sd = dt.datetime(2008, 1, 1)
    in_ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    sv = 100000
    commission = 0
    impacts = [0, 0.0025, 0.005, 0.01]

    output = experiment2(in_sd, in_ed, symbol, sv, commission, impacts)
    print(output)

def author():
    return 'jchen384'