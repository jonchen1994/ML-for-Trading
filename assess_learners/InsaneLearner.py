import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.model_list = []
    def author(self): return "jchen384"
    def add_evidence(self, data_x, data_y):
        for x in range(0, 20):
            model = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
            model.add_evidence(data_x, data_y)
            self.model_list.append(model)
    def query(self, points):
        outputs = []
        for x in self.model_list:
            outputs.append(x.query(points))
        return np.array(outputs).mean(axis=0)


