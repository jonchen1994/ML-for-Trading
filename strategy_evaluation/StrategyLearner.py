""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Jonathan Chen (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: jchen384 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 902912657 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt
import random
import numpy as np
import pandas as pd
import util as ut
import QLearner as ql
from indicators import MACD, EMA, momentum, SMA, bollingerBand
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission

    def author(self):
        return 'jchen384'
    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		   	 		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        sv=10000,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # example usage of the old backward compatible util function
        pd.set_option('display.max_rows', None)
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        boll = self.get_bollinger(symbol, sd, ed, n = 3)
        macd = self.get_MACD(symbol, sd, ed, n = 3, quick = 30, slow = 16, signal = 8)
        mom = self.get_momentum(symbol, sd, ed, n = 3, window = 15)

        prices['Boll'] = boll.values
        prices['Mac-Diff'] = macd['Difference'].values
        prices['Mac-Sig'] = macd['Signal'].values
        prices['Momentum'] = mom.values

        indicators = prices[['Boll','Mac-Diff','Mac-Sig','Momentum']]
        self.states = self.descretize_state(indicators)

        self.learner = ql.QLearner(
            num_states=len(self.states),
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.90,
            radr=0.999,
            dyna=0,
            verbose=False,
        )

        max_iterations = 210
        prev_result = 0
        count = 0
        curr_result = 1
        all_ret = []
        # print(prices.head(10))
        while count < max_iterations and prev_result != curr_result:
            prev_result = curr_result
            start_ind = prices.iloc[0]
            curr_pos = 1
            enc_state = self.make_state(start_ind['Boll'], start_ind['Mac-Diff'], start_ind['Mac-Sig'], start_ind['Momentum'])
            state = self.get_state_ind(self.states, enc_state)
            action = self.learner.querysetstate(state)
            cash = sv
            holdings_val = 0
            old_val = cash + holdings_val
            trades = []
            returns = []
            posits = []
            for i in range(0, len(prices)):
                row = prices.iloc[i]
                posits.append(curr_pos)
                # curr_pos = int(states[state]/10000)
                curr_pos, cash, holdings_val = self.new_position(curr_pos, action, row[symbol], cash, holdings_val)
                #print("s: " + str(curr_pos) + " a: " + str(action) + " s': " + str(pos) + " cash: " + str(cash) + " holding_val: " + str(holdings_val) + " price: " + str(row[syms].values))
                new_val = cash + holdings_val
                ret = ((new_val - old_val)/old_val)
                returns.append(ret)
                old_val = new_val
                enc_state = self.make_state(row['Boll'], row['Mac-Diff'], row['Mac-Sig'], row['Momentum'])
                state = self.get_state_ind(self.states, enc_state)
                action = self.learner.query(state, ret)
                trades.append(action)

            curr_result = trades
            count +=1



    def new_position(self, curr_pos, action, curr_price, cash, holdings_val):
        action = action -1
        curr_pos = curr_pos -1
        if (curr_pos == -1 and action == -1) or (curr_pos == -1 and action == 0):   # You are shorting and then you either short, or do nothing. Your position doens't change update holding_val
            pos = curr_pos
            holdings_val = (-1) * curr_price * 1000
        elif (curr_pos == 0 and action == -1):              # You are cashed out and want to short You get the cash You take the short position Your holdings become the short equivalent
            pos = -1
            holdings_val = (-1) * curr_price * 1000
            cash = cash + (curr_price * 1000) - (self.impact * abs(holdings_val)) - self.commission
        elif (curr_pos == 1 and action == -1):                      # You are long and want to short You get all the cash You take the short position Your holdings become the short equivalent
            holdings_val = (-1) * curr_price * 1000
            cash = cash + (curr_price * 2000) - (self.impact * abs(2*holdings_val)) - self.commission
            pos = -1
        elif (curr_pos == 0 and action == 0):               #You are cashed out and you do nothing. No change in position.
            pos = curr_pos
        elif (curr_pos == 1 and action == 0) or (curr_pos == 1 and action == 1):    #You are long and want to buy or do nothing Your position doens't change Price is updated
            pos = curr_pos
            holdings_val = curr_price * 1000
        elif (curr_pos == -1 and action == 1):      #You are shorting and want to go long If you have the cash Update state to Buy Deduct cash Update Holdings
            if cash >= curr_price * 2000:
                pos = 1
                holdings_val = curr_price * 1000
                cash -= curr_price * 2000
                cash -= (self.impact * abs(2*holdings_val)) - self.commission
            else: #You are unable to buy and forced to hold (MAYBE ADD OPTION TO CASh OUT) You are still shorting - update the holdings value
                pos = curr_pos
                holdings_val = (-1) * curr_price * 1000
        elif (curr_pos == 0 and action == 1):       #You are cashed out and want to buy Check if you can buy 1000 Change position to Long Reflect Change in cash Reflect Change in holdings.
            if cash >= curr_price * 1000:
                pos = 1
                holdings_val = curr_price * 1000
                cash -= curr_price * 1000
                cash -= (self.impact * abs(holdings_val)) - self.commission
            else:
                pos = curr_pos                      #your position doens't chagne

        else:
            pass

        return pos+1, cash, holdings_val

    def descretize_ind(self,input, bins = 4):
        df = input.copy()
        z_d = 0 - df.min()
        df = df + z_d
        df = df / df.max() * (bins-1)
        df = df.round()
        return df

    def descretize_state(self, indicators):
        mom_values = indicators['Momentum'].unique()
        macd_values = indicators['Mac-Diff'].unique()
        macsig_values = indicators['Mac-Sig'].unique()
        boll_values = indicators['Boll'].unique()

        states = []
        for mom in mom_values:
            for macd in macd_values:
                for macsig in macsig_values:
                    for bll in boll_values:
                        states.append(self.make_state(mom,macd,macsig,bll))

        return states

    def get_state_ind(self, states, current_state):
        return states.index(current_state)

    def make_state(self, bll, macd, macsig,  mom):
        return int((bll*1000) + (macd*100) + (macsig*10) + mom)

    def get_bollinger(self, symbol, sd, ed, n = 0, window=20, width=2):
        boll_sd = sd - dt.timedelta(days=window * 2)
        prices_boll = ut.get_data([symbol], pd.date_range(boll_sd, ed))
        prices_boll = prices_boll[symbol]
        prices_boll = prices_boll / prices_boll.iloc[0]
        boll = bollingerBand(prices_boll, window, width)
        if n > 0:
            sd = boll[:sd].tail(n).index[0]
            ed = boll[:ed].tail(n+1).index[0]
        boll = boll[sd:ed]
        df = self.descretize_ind(boll)
        return df

    def get_momentum(self, symbol, sd, ed, n = 0, window=5):
        mom_sd = sd - dt.timedelta(days=window * 3)
        prices_mom = ut.get_data([symbol], pd.date_range(mom_sd, ed))
        prices_mom = prices_mom[symbol]
        mom = momentum(prices_mom, window)
        if n > 0:
            sd = mom[:sd].tail(n).index[0]
            ed = mom[:ed].tail(n+1).index[0]
        mom = mom[sd:ed]
        df = self.descretize_ind(mom)
        return df

    def get_MACD(self, symbol, sd, ed, n = 0, quick=26, slow=12, signal=9):
        mac_sd = sd - dt.timedelta(days=quick * 2)
        prices_mac = ut.get_data([symbol], pd.date_range(mac_sd, ed))
        prices_mom = prices_mac[symbol]
        mac = MACD(prices_mom, quick, slow, signal)
        if n > 0:
            sd = mac[:sd].tail(n).index[0]
            ed = mac[:ed].tail(n+1).index[0]
        mac = mac[sd:ed]
        df = self.descretize_ind(mac)
        return df

    def n_date_delta(self, sd, ed, n = 0):
        new_sd = sd - dt.timedelta(days=n)
        new_ed = ed - dt.timedelta(days=n)
        return new_sd, new_ed



    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        Tests your learner using data outside of the training data
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        boll = self.get_bollinger(symbol, sd, ed, n=3)
        macd = self.get_MACD(symbol, sd, ed, n=3, quick=30, slow=16, signal=8)
        mom = self.get_momentum(symbol, sd, ed, n=3, window=15)

        prices['Boll'] = boll.values
        prices['Mac-Diff'] = macd['Difference'].values
        prices['Mac-Sig'] = macd['Signal'].values
        prices['Momentum'] = mom.values

        start_ind = prices.iloc[0]
        curr_pos = 1
        enc_state = self.make_state(start_ind['Boll'], start_ind['Mac-Diff'], start_ind['Mac-Sig'],
                                    start_ind['Momentum'])
        state = self.get_state_ind(self.states, enc_state)
        action = self.learner.querysetstate(state)
        cash = sv
        holdings_val = 0
        trades = []
        for i in range(0, len(prices)):
            row = prices.iloc[i]
            curr_pos, cash, holdings_val = self.new_position(curr_pos, action, row[symbol], cash, holdings_val)
            # print("s: " + str(curr_pos) + " a: " + str(action) + " s': " + str(pos) + " cash: " + str(cash) + " holding_val: " + str(holdings_val) + " price: " + str(row[syms].values))
            enc_state = self.make_state(row['Boll'], row['Mac-Diff'], row['Mac-Sig'],row['Momentum'])

            state = self.get_state_ind(self.states, enc_state)
            action = self.learner.querysetstate(state)
            trades.append(action - 1)


        df_trades = pd.DataFrame(index=prices.index)
        df_trades['Actions'] = trades
        # print(trades)
        df_trades = self.get_orders(df_trades)
        return df_trades

    def get_orders(self, actions):
        trades = actions['Actions'] * 2000

        z = [i for i, x in enumerate(trades) if x != 0]
        trades[z[0]] = trades[z[0]] * .5

        position = 0
        for t in range(len(trades)):
            new_pos = position + trades[t]
            if new_pos != 0 and new_pos != 1000 and new_pos != -1000:
                trades[t] = 0
            position = position + trades[t]

        orders = pd.DataFrame(index=actions.index)
        orders['Shares'] = trades
        return orders

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("One does not simply think up a strategy")  		  	   		   	 		  		  		    	 		 		   		 		  
