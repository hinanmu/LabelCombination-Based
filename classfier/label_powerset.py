#@Time      :2018/12/3 20:27
#@Author    :zhounan
# @FileName: label_powerset.py
import numpy as np

class LabelPowerset():
    def __init__(self, classfier):
        self.classfier = classfier
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.label_count = None

    def fit(self, X, y):
        self.classfier.fit(X, self.trainsform(y))

        return self

    def predict(self, X):
        lp_prediction = self.classifier.predict(X)

        return self.inverse_transform(lp_prediction)

    def predict_proba(self,X):
        lp_prediction = self.classifier.predict_proba(X)
        result = np.array((X.shape[0], self.label_count))
        for row in range(len(lp_prediction)):
            assignment = lp_prediction[row]
            for combination_id in range(len(assignment)):
                for label in self.reverse_combinations[combination_id]:
                    result[row, label] += assignment[combination_id]

        return result

    def trainsform(self, y):
        self.label_count = y.shape[1]
        last_id = 0
        y_class = []
        for y_i in y:
            label_string = ",".join(map(str, y_i))

            if label_string not in self.unique_combinations:
                self.unique_combinations_[label_string] = last_id
                self.reverse_combinations.append(y_i)
                last_id += 1

            y_class.append(self.unique_combinations[label_string])

        return np.array(y_class)

    def inverse_trainsform(self, y):
        n_samples = len(y)
        result = np.array((n_samples, self.label_count), dtype=np.int32)
        for row in range(n_samples):
            assignment = y[row]
            result[row] = self.reverse_combinations[assignment]

        return result