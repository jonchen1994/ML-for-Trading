import numpy as np


class RTLearner(object):


    def __init__(self, leaf_size = 1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass

    def author(self):
        return "jchen384"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        output = self.build_tree(data_x,data_y)

        # build and save the model
        self.model_tree = output

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size: return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if np.all([data_y[0] == x for x in data_y]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])
        else:
            index = self.best_feature(data_x, data_y)
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
        return np.random.randint(0,data_x.shape[1])

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
