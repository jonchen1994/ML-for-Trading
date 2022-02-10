""""""
"""Assess a betting strategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Jonathan Chen 		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: jchen384	  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 902912657	   		   	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return "jchen384"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return 902912657  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		   	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def strategy(win_prob, win_threshold = 80, bet_limit = 1000):
    track_winnings = np.empty(bet_limit+1)
    track_winnings[:] = np.nan

    episode_winnings = 0
    track_winnings[0] = episode_winnings
    index = 1
    while episode_winnings < win_threshold and index <= bet_limit:
        won = False
        bet_amount = 1
        while not won and index <= bet_limit:
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = bet_amount * 2
            track_winnings[index] = episode_winnings
            index += 1

    return track_winnings


def strategyR(win_prob, win_threshold = 80, bet_limit = 1000, floor = -256):
    track_winnings = np.empty(bet_limit+1)
    track_winnings[:] = np.nan

    episode_winnings = 0
    track_winnings[0] = episode_winnings
    index = 1
    while episode_winnings < win_threshold and index <= bet_limit and episode_winnings > floor:
        won = False
        bet_amount = 1
        while not won and index <= bet_limit and episode_winnings > floor:
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = bet_amount * 2 if episode_winnings - (
                            2 * bet_amount) >= floor else episode_winnings - floor

            track_winnings[index] = episode_winnings
            index += 1

    return track_winnings


def episodes_accum(num_episodes, win_prob, bet_limit = 1000, real=False):
    all = np.empty((bet_limit + 1, num_episodes))
    if not real:
        for x in range(num_episodes):
            all[:, x] = strategy(win_prob, bet_limit=bet_limit)
    else:
        for x in range(num_episodes):
            all[:, x] = strategyR(win_prob, bet_limit=bet_limit)
    return all

def graphingEpisodes(df, title, figsize = (14,8), xlims = [0,300], ylims = [-256,100]):
    plt.figure(figsize=figsize)
    plt.plot(df)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("Number of Bets")
    plt.ylabel("Earnings")
    plt.title(title)
    plt.legend(df.columns, loc='upper right')
    plt.savefig(f'{title}.png')


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    win_prob = 0.474  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		   	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		   	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments

    #Exp 1.1

    # num_episodes = 10
    # df = episodes_accum(num_episodes,win_prob)
    # df = pd.DataFrame(df, columns=range(1, num_episodes + 1))
    # df = df.fillna(method='ffill', axis=0)
    # graphingEpisodes(df, title = "10 Episodes")

    # Exp 1.2 + 1.3

    # num_episodes = 1000
    # df = episodes_accum(num_episodes, win_prob)
    # df = pd.DataFrame(df, columns=range(1, num_episodes + 1))
    # df = df.fillna(method='ffill', axis=0)
    #
    # stats = pd.DataFrame()
    #
    # stats['Mean'] = df.mean(axis=1)
    # stats['Med'] = df.median(axis=1)
    # stats['Std'] = df.std(axis=1)
    # stats['UpperMean'] = stats['Mean'] + stats['Std']
    # stats['LowerMean'] = stats['Mean'] - stats['Std']
    # stats['UpperMed'] = stats['Med'] + stats['Std']
    # stats['LowerMed'] = stats['Med'] - stats['Std']

    # #1.2
    # graphingEpisodes(stats[['Mean', 'UpperMean', 'LowerMean']], title = "Mean of 1000 Episodes with Std")
    # #1.3
    # graphingEpisodes(stats[['Med', 'UpperMed', 'LowerMed']], title = "Median of 1000 Episodes with Std")
    #
    # #2.1
    num_episodes = 1000
    df = episodes_accum(num_episodes, win_prob, real=True)
    df = pd.DataFrame(df, columns=range(1, num_episodes + 1))
    df = df.fillna(method='ffill', axis=0)

    stats = pd.DataFrame()

    stats['Mean'] = df.mean(axis=1)
    stats['Med'] = df.median(axis=1)
    stats['Std'] = df.std(axis=1)
    stats['UpperMean'] = stats['Mean'] + stats['Std']
    stats['LowerMean'] = stats['Mean'] - stats['Std']
    stats['UpperMed'] = stats['Med'] + stats['Std']
    stats['LowerMed'] = stats['Med'] - stats['Std']

    print(stats['UpperMean'].max())
    print(stats['LowerMean'].min())
    #
    # #2.1
    # graphingEpisodes(stats[['Mean', 'UpperMean', 'LowerMean']], title="Realistic Mean of 1000 Episodes with Std")
    # #2.2
    # graphingEpisodes(stats[['Med', 'UpperMed', 'LowerMed']], title="Realistic Median of 1000 Episodes with Std")

if __name__ == "__main__":
    test_code()
