""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
import warnings  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		   	 		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose=False):  		  	   		   	 		  		  		    	 		 		   		 		  

        self.leaf_size = leaf_size
        self.verbose = verbose
        pass
  		  	   		   	 		  		  		    	 		 		   		 		  
    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "jchen384"  # replace tb34 with your Georgia Tech username
  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        output = self.build_tree(data_x, data_y)

        # build and save the model
        self.model_tree = output

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size: return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if np.all([data_y[0] == x for x in data_y]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        else:
            index = self.best_feature(data_x, data_y)

            if index is np.nan:
                print(index)
                return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
            else:
                median = np.median(data_x[:, index])
                mask = data_x[:, index] <= median

                if sum(mask) == 0 or sum(mask) == data_x.shape[0]:
                    median = np.mean(data_x[:, index])
                    mask = data_x[:, index] <= median

                lefttree_x = np.array([data_x[x] for x in range(data_x.shape[0]) if mask[x]])
                lefttree_y = np.array([data_y[x] for x in range(data_x.shape[0]) if mask[x]])
                lefttree = self.build_tree(lefttree_x, lefttree_y)
                righttree_x = np.array([data_x[x] for x in range(data_x.shape[0]) if not mask[x]])
                righttree_y = np.array([data_y[x] for x in range(data_x.shape[0]) if not mask[x]])
                righttree = self.build_tree(righttree_x, righttree_y)
                root = np.array([[index, median, 1, lefttree.shape[0] + 1]])

                return np.vstack((root, lefttree, righttree))

    def best_feature(self, data_x, data_y):
        corrs = [abs(np.corrcoef(data_x[:, x], data_y)[0, 1]) for x in range(data_x.shape[1])]


        try:
            return np.nanargmax(corrs)
        except ValueError:
            return np.nan

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        output = []
        for x in points:
            index = 0
            branch = self.model_tree[index]
            while branch[0] != -1:
                if x[int(branch[0])] <= branch[1]:
                    index += branch[2]
                else:
                    index += branch[3]

                branch = self.model_tree[int(index)]

            output.append(branch[1])
        return output
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		   	 		  		  		    	 		 		   		 		  
