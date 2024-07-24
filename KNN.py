import numpy as np

class KNearestNeighbors:
    def init(self, k=1, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        distances = self._calc_distances(X_test)
        k_nearest_labels, k_nearest_distances = self._find_k_nearest_labels(distances)
        return self._predict_labels(k_nearest_labels, k_nearest_distances)

    def _calc_distances(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                if self.metric == 'euclidean':
                    distances[i][j] = np.sqrt(np.sum((X_test[i] - self.X_train[j]) ** 2))
                elif self.metric == 'manhattan':
                    distances[i][j] = np.sum(np.abs(X_test[i] - self.X_train[j]))
                else:
                    raise ValueError(f"Unsupported metric: {self.metric}")
        return distances

    def _find_k_nearest_labels(self, distances):
        num_test = distances.shape[0]
        k_nearest_labels = np.zeros((num_test, self.k))
        k_nearest_distances = np.zeros((num_test, self.k))
        for i in range(num_test):
            k_nearest_indices = distances[i].argsort()[:self.k]
            k_nearest_labels[i] = self.y_train[k_nearest_indices]
            k_nearest_distances[i] = distances[i, k_nearest_indices]
        return k_nearest_labels, k_nearest_distances

    def _predict_labels(self, k_nearest_labels, k_nearest_distances):
        num_test = k_nearest_labels.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            if self.weights == 'uniform':
                unique_labels, counts = np.unique(k_nearest_labels[i], return_counts=True)
                y_pred[i] = unique_labels[np.argmax(counts)]
            elif self.weights == 'distance':
                unique_labels = np.unique(k_nearest_labels[i])
                label_weights = {label: 0 for label in unique_labels}
                for j, label in enumerate(k_nearest_labels[i]):
                    label_weights[label] += 1 / (k_nearest_distances[i, j] + 1e-6)
                y_pred[i] = max(label_weights, key=label_weights.get)
            else:
                raise ValueError(f"Unsupported weights type: {self.weights}")
        return y_pred