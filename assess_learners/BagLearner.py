import numpy as np
from RTLearner import RTLearner
import LinRegLearner as lrl
from DTLearner import DTLearner


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.verbose = verbose
        self.tricks = []
        pass

    def author(self):
        return "jchen384"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        for x in range(0, self.bags):
            model = self.learner(**self.kwargs)
            bag_x, bag_y = self.bag_data(data_x, data_y)
            model.add_evidence(bag_x, bag_y)
            self.tricks.append(model)

    def bag_data(self, data_x, data_y):
        index = np.random.choice(list(range(0, data_x.shape[0])), data_x.shape[0])
        return data_x[index], data_y[index]

    def query(self, points):
        outputs = []
        for x in self.tricks:
            outputs.append(x.query(points))

        return np.array(outputs).mean(axis=0)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
