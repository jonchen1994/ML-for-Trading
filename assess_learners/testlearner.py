""""""
import DTLearner

"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
"""
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import sys
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as dl
import random
import BagLearner as bl
import matplotlib.pyplot as plt
import RTLearner as rl
import time

if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		   	 		  		  		    	 		 		   		 		  
        sys.exit(1)

    inf = open(sys.argv[1])


    data = np.array(
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
    )

    random.seed(902912657)

    #Random sampling

    mask = list(range(0,data.shape[0]))
    train = random.sample(mask, int(len(mask)*.6))

    train_x = np.array([data[x, 0:-1] for x in range(0,data.shape[0]) if x in train])
    train_y = np.array([data[x, -1] for x in range(0, data.shape[0]) if x in train])

    test_x = np.array([data[x, 0:-1] for x in range(0, data.shape[0]) if x not in train])
    test_y = np.array([data[x, -1] for x in range(0, data.shape[0]) if x not in train])


    #a) DTLearner
    dPerf= []
    for x in range(1,15):
        temp = []
        d = dl.DTLearner(leaf_size=x)
        d.add_evidence(train_x,train_y)
        pred_train = d.query(train_x)
        pred_test = d.query(test_x)
        temp.append(math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0]))
        temp.append(math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0]))
        dPerf.append(temp)

    plt.plot(np.array(dPerf))
    plt.legend(["training", "test"], loc='upper right')
    plt.xlabel("Leaf Size")
    plt.xlim([1, 14])
    plt.ylim([0, .01])
    plt.ylabel("RMSE")
    plt.title("Training vs Testing Error")
    plt.savefig('Experiment1.png')
    plt.clf()

    #b) Bagging
    dPerf = []
    for x in range(1, 15):
        temp = []
        b = bl.BagLearner(learner = dl.DTLearner, kwargs = {"leaf_size":x}, bags = 15, boost = False, verbose = False)
        b.add_evidence(train_x, train_y)
        pred_train = b.query(train_x)
        pred_test = b.query(test_x)
        temp.append(math.sqrt(((train_y - pred_train) ** 2).sum() / train_y.shape[0]))
        temp.append(math.sqrt(((test_y - pred_test) ** 2).sum() / test_y.shape[0]))
        dPerf.append(temp)

    plt.plot(np.array(dPerf))
    plt.legend(["training", "test"], loc='upper right')
    plt.xlabel("Leaf Size")
    plt.xlim([1, 14])
    plt.ylim([0, .01])
    plt.ylabel("RMSE")
    plt.title("Bagged Training vs Testing Error")
    plt.savefig('Experiment2.png')
    plt.clf()

    ####3
    times = []
    acc = []
    for x in range(1,15):
        tempT = []
        tempA = []
        t1 = time.clock()
        d = dl.DTLearner(leaf_size=x)
        d.add_evidence(train_x, train_y)
        t2 = time.clock()
        r = rl.RTLearner(leaf_size=x)
        r.add_evidence(train_x, train_y)
        t3 = time.clock()

        d_pred = d.query(test_x)
        r_pred = r.query(test_x)

        tempA.append(np.mean((abs(test_y - d_pred) / test_y))*100)
        tempA.append(np.mean((abs(test_y - r_pred) / test_y))*100)
        acc.append(tempA)

        tempT.append(abs(t2 - t1))
        tempT.append(abs(t3 - t2))
        times.append(tempT)


    plt.plot(np.array(acc))
    plt.legend(["Decision", "Random"], loc='upper right')
    plt.xlabel("Leaf Size")
    plt.xlim([1, 14])
    plt.ylabel("MAPE in %'s ")
    plt.title("Decision vs. Random MAPE")
    plt.savefig('Experiment3a.png')
    plt.clf()

    plt.plot(np.array(times))
    plt.legend(["Decision", "Random"], loc='upper right')
    plt.xlabel("Leaf Size")
    plt.xlim([1, 14])
    plt.ylabel("Time")
    plt.title("Decision vs. Random Training Time")
    plt.savefig('Experiment3b.png')
    plt.clf()

