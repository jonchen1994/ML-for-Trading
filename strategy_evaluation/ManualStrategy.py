""""""
import numpy as np

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

import pandas as pd
from util import get_data, plot_data
from indicators import MACD, EMA, momentum, SMA, bollingerBand
import marketsimcode as mktsim


def testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):

    """
    1) Get Data
    2) Run Data through Strategy
        a) Strategy should return signals based on dates and signals
        b) Get action/date tuples
    3) Evaluate
    """

    boll = get_bollinger(symbol, sd, ed)
    bol_acts = boll_vote(boll)
    mac = get_MACD(symbol, sd, ed, 30, 16, 8)
    mac_acts = mac_vote(mac)
    mom = get_momentum(symbol, sd, ed, 15)
    mom_acts = mom_vote(mom)

    strat = strategy(bol_acts, mom_acts, mac_acts)
    return get_orders(strat)


# def get_SMA(symbol, sd, ed, window=5):
#     sma_sd = sd - dt.timedelta(days=window * 2)
#     prices_sma = get_data([symbol], pd.date_range(sma_sd, ed))
#     prices_sma = prices_sma[symbol]
#     sma = SMA(prices_sma, window)[sd:ed]
#     return sma


def get_bollinger(symbol, sd, ed, window=20, width=2):
    boll_sd = sd - dt.timedelta(days=window * 2)
    prices_boll = get_data([symbol], pd.date_range(boll_sd, ed))
    prices_boll = prices_boll[symbol]
    prices_boll = prices_boll / prices_boll.iloc[0]

    boll = bollingerBand(prices_boll, window, width)[sd:ed]
    return boll


def boll_vote(bollinger):
    b_state = 0
    prev_b = 0
    actions = []
    for b in bollinger:
        if prev_b > 1 and b_state != -1:
            if prev_b > b and b < .92:
                b_state = -1
                actions.append(-2)
            else:
                actions.append(0)
        elif b_state == -1 and b <= .2:
            b_state = 0
            actions.append(1)
        elif prev_b < -1 and b_state != 1:
            if prev_b < b and b > -.92:
                b_state = 1
                actions.append(2)
            else:
                actions.append(0)
        elif b_state == 1 and b >= -.2:
            b_state = 0
            actions.append(-1)
        else:
            actions.append(0)

        prev_b = b

    df = pd.DataFrame(index=bollinger.index)
    df['boll_action'] = actions
    return df


def get_momentum(symbol, sd, ed, window=5):
    mom_sd = sd - dt.timedelta(days=window * 2)
    prices_mom = get_data([symbol], pd.date_range(mom_sd, ed))
    prices_mom = prices_mom[symbol]
    mom = momentum(prices_mom, window)[sd:ed]
    return mom


def mom_vote(momentum, threshold=5):
    mom_overall = 0
    prev_mom = 0
    mom_ovs = []
    mom_actions = []
    for m in momentum:
        if m > 0:
            mom_overall += 1
        else:
            mom_overall -= 1

        mom_ovs.append(mom_overall)

        if prev_mom <= 0 and m > 0:
            mom_actions.append(2)  # Pure buy signal.
        elif prev_mom >= 0 and m < 0:
            mom_actions.append(-2)  # Pure sell
        else:
            mom_actions.append(0)

        prev_mom = m

    df = pd.DataFrame(index=momentum.index)
    df['mom_streak'] = mom_ovs
    df['mom_action'] = mom_actions
    return df


def get_MACD(symbol, sd, ed, quick=26, slow=12, signal=9):
    mac_sd = sd - dt.timedelta(days=quick * 2)
    prices_mac = get_data([symbol], pd.date_range(mac_sd, ed))
    prices_mom = prices_mac[symbol]
    mac = MACD(prices_mom, quick, slow, signal)[sd:ed]
    return mac


def mac_vote(macd):
    # If before prev_mac was larger than prev_sig and then they cross, then we sell.
    # If before prev sig was larger than prev_mac and then they cross, then we buy.
    prev_mac = 0
    prev_sig = 0
    d_vote = []
    mac_state = 0
    for index, row in macd.iterrows():
        mac = row['Difference']
        sig = row['Signal']
        if prev_mac > prev_sig:
            if mac < sig:
                d_vote.append(-2)
            else:
                d_vote.append(0)
        elif prev_mac < prev_sig:
            if mac > sig:
                d_vote.append(2)
            else:
                d_vote.append(0)
        else:
            d_vote.append(0)

        prev_mac = mac
        prev_sig = sig

    df = pd.DataFrame(index=macd.index)
    df['macd_action'] = d_vote
    return df


def get_orders(actions):
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


def strategy(boll_votes, mom_votes, mac_votes):
    together = pd.concat([boll_votes, mom_votes, mac_votes], axis=1)
    together.columns = ['boll_action', 'mom_streak', 'mom_action', 'macd_action']
    strat = []
    for index, rows in together.iterrows():
        b_a = rows['boll_action']
        m_s = rows['mom_streak']
        m_a = rows['mom_action']
        d_a = rows['macd_action']

        if b_a != 0:
            act = b_a
        elif m_s * m_a > 0 and abs(m_s) > 20:
            act = m_a
        elif abs(m_s) < 15:
            act = d_a
        else:
            act = 0

        if act > 0:
            strat.append(1)
        elif act < 0:
            strat.append(-1)
        else:
            strat.append(0)

    df = pd.DataFrame(index=boll_votes.index)
    df['Actions'] = strat
    return df

def author():
    return 'jchen384'

if __name__ == "__main__":
    testPolicy(symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv=100000)
