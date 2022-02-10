""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import random as rand  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		   	 		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		   	 		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		   	 		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		   	 		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		   	 		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		   	 		  		  		    	 		 		   		 		  
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.num_actions = num_actions
        self.num_states = num_states
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((num_states, num_actions))
        # self.Tc = np.zeros((num_states, num_actions, num_states)) + .00001
        self.Tc = []
        self.R = np.zeros((num_states, num_actions))
        self.s = 0
        self.a = 0
        self.dyna = dyna

  		  	   		   	 		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		   	 		  		  		    	 		 		   		 		  
        :type s: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        if rand.random() <= self.rar:
            self.s = s
            action = rand.randint(0, self.num_actions - 1)
            self.a = action
            #print("random action " + str(action))
            return action
        else:
            action = self.Q[self.s].argmax()
            self.s = s
            self.a = action
            #print("regular action " + str(action))
            return action

    def query(self, s_prime, r):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		   	 		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		   	 		  		  		    	 		 		   		 		  
        :type r: float  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        # reward = self.alpha * (r + (self.gamma * self.Q[s_prime].max()))  # Change
        # self.Q[self.s, self.a] = ((1 - self.alpha) * self.Q[self.s, self.a]) + reward
        self.update_q(self.s, self.a, s_prime, r)
        self.update_r(self.s, self.a, r)
        self.Tc.append([self.s, self.a, s_prime])
        # self.update_tc(self.s, self.a, s_prime)
        #print(self.Q)

        if rand.random() <= self.rar:
            self.rar = self.rar * self.radr
            self.s = s_prime
            action = rand.randint(0, self.num_actions - 1)
            self.a = action
            #print("random action " + str(action))
            self.run_dyna()
            return action
        else:
            action = self.Q[s_prime].argmax()
            self.s = s_prime
            self.a = action
            #print("regular action " + str(action))
            self.run_dyna()
            return action



    def run_dyna(self):
        for d in range(self.dyna):
            # Hallucinate
            # non1 = self.Q.sum(axis=1).nonzero()[0] #List of states that have been visited
            # r_s = non1[rand.randint(0, len(non1) - 1)]
            # non2 = self.Q[r_s].nonzero()[0] #List of actions that have been visited
            # r_a = non2[rand.randint(0, len(non2) - 1)]
            # prob = self.Tc[r_s, r_a] - .00001
            # s_pr = rand.choices(list(range(0, self.num_states)), weights=prob/sum(prob))
            exp = rand.choice(self.Tc)
            r = self.R[exp[0], exp[1]]

            # Update Q
            self.update_q(exp[0], exp[1], exp[2], r)

    def update_q(self, s, a, s_pr, r):
        reward = self.alpha * (r + (self.gamma * self.Q[s_pr].max()))  # Change
        self.Q[s, a] = ((1 - self.alpha) * self.Q[s, a]) + reward

    # def update_tc(self, s, a, s_pr):
    #     #print("state: "+ str(s)+" action: " + str(a) + " state_prime:" + str(s_pr))
    #     self.Tc[s, a, s_pr] += 1

    def update_r(self, s, a, r):
        self.R[s, a] = ((1-self.alpha) * self.R[s, a]) + (self.alpha * r)

    def author(self):
        return "jchen384"


if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		   	 		  		  		    	 		 		   		 		  
