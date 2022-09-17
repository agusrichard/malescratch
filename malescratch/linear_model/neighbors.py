import numpy as np
import math
import MinMaxScaler, StandardScaler, accuracy_score, r2_score


class KNNBase(object):
    def __init__(self, k_neighbors, scaler, order):
        self.k_neighbors = k_neighbors
        self.scaler = scaler
        self.order = order

    def fit(self, X, y):
        if self.scaler == None:
            self._X = X
        elif self.scaler == "minmax":
            self._scaler = MinMaxScaler()
            self._X = self._scaler.fit_transform(X)
        elif self.scaler == "standard":
            self._scaler = StandardScaler()
            self._X = self._scaler.fit_transform(X)
        else:
            raise ValueError("Unknown scaler")

        self._y = y
        return self

    def predict():
        pass

    def score():
        pass


class KNNClassifier(KNNBase):
    def __init__(self, k_neighbors=5, scaler=None, order=2):
        super().__init__(k_neighbors=k_neighbors, scaler=scaler, order=order)

    def _vote(self, arg_sorted, y_fit):
        result = np.argmax(np.bincount(y_fit[arg_sorted]).astype("int"))
        proba = np.bincount(y_fit[arg_sorted], minlength=len(np.unique(y_fit))) / len(
            np.unique(y_fit)
        )
        return result, proba

    def predict(self, X):
        if self.scaler == None:
            X_pred = X
        else:
            X_pred = self._scaler.transform(X)
        X_fit = self._X
        y_fit = self._y

        percentage = 0
        predictions = np.zeros(X_pred.shape[0], dtype="int")
        predict_proba = np.zeros(shape=(X_pred.shape[0], np.unique(y_fit).shape[0]))
        for i in range(predictions.shape[0]):
            distance = np.zeros(X_fit.shape[0])
            for j in range(distance.shape[0]):
                distance[j] = np.linalg.norm(X_pred[i] - X_fit[j], self.order)

            arg_sorted = np.argsort(distance)[: self.k_neighbors]
            predictions[i], predict_proba[i] = self._vote(arg_sorted, y_fit)
            if i % (predictions.shape[0] / 100) == 0:
                print("\rProcess {}%".format(percentage), end="")
                percentage += 1
        print("\rProcess 100%", end="\n")

        self._predictions = predictions
        return predictions, predict_proba

    def score(self, X, y):
        try:
            predictions = self._predictions
        except Exception:
            predictions, _ = self.predict(X)
        labels = y

        return accuracy_score(predictions, labels)


class KNNRegressor(KNNBase):
    def __init__(self, k_neighbors=5, scaler=None, order=2, aggregate="mean"):
        self.aggregate = aggregate
        super().__init__(k_neighbors=k_neighbors, scaler=scaler, order=order)

    def _aggregate(self, arg_sorted, y_fit):
        # print("Nearesst neighbors: ", y_fit[arg_sorted])
        if self.aggregate == "mean":
            result = np.mean(y_fit[arg_sorted])
        elif self.aggregate == "median":
            result = np.median(y_fit[arg_sorted])
        else:
            raise ValueError("Unknown aggregate")
        # print("Result: ", result)
        return result

    def predict(self, X):
        if self.scaler == None:
            X_pred = X
        else:
            X_pred = self._scaler.transform(X)
        X_fit = self._X
        y_fit = self._y

        percentage = 0
        predictions = np.zeros(X_pred.shape[0], dtype="float")
        for i in range(predictions.shape[0]):

            distance = np.zeros(X_fit.shape[0], dtype="float")
            for j in range(distance.shape[0]):
                distance[j] = np.linalg.norm(X_pred[i] - X_fit[j], self.order)

            arg_sorted = np.argsort(distance)[: self.k_neighbors]
            # print(arg_sorted)
            predictions[i] = self._aggregate(arg_sorted, y_fit)

            if i % (predictions.shape[0] / 100) == 0:
                print("\rProcess {}%".format(percentage), end="")
                percentage += 1
        print("\rProcess 100%", end="\n")

        self._predictions = predictions
        return predictions

    def score(self, X, y):
        try:
            predictions = self._predictions
        except Exception:
            predictions, _ = self.predict(X)
        labels = y

        return r2_score(predictions, labels)
