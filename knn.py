from scipy.spatial import distance
from collections import Counter
import numpy as np 

class KNearestNeighbours:
    def __init__(self,k=5,distance_method='euclidean'):
        self.neighbour_count = k
        self.distance_method = distance_method
        self._check_distance_method_name(self.distance_method)
        self._check_neighbor_count(self.neighbour_count)

    def _check_neighbor_count(self):
        if self.neighbour_count <1:
            raise Exception("Neighbour count must be bigger than 0.")

    def _check_distance_method_name(self):
        methods = ['euclidean','minkowski','manhattan','chebyshev','jaccard','cosine']
        if self.distance_method not in methods:
            raise Exception(f"Undefined distance method is given. Given method: {self.distance_method}. Method can be defined {methods}")

    def fit(self,x:np.array,y:np.array):
        self.X_train = x
        self.y_train = y
        
    def __calculate_distance(self,u:np.array,v:np.array) -> float:
        methods = {
            'euclidean':distance.euclidean(u,v),
            'minkowski':distance.minkowski(u,v),
            'manhattan':distance.cityblock(u,v),
            'chebyshev':distance.chebyshev(u,v),
            'jaccard':distance.jaccard(u,v),
            'cosine': distance.cosine(u,v),
            }
        return methods.get(self.distance_method,None)

    def __predict(self,x_pred:np.array):
        distances = [self.__calculate_distance(x_pred,x_real) for x_real in self.X_train]
        sorted_distances_as_idx = np.argsort(distances)
        knn_indices = sorted_distances_as_idx[:self.neighbour_count]
        predicted_values = self.y_train[knn_indices].squeeze().tolist()
        most_common_values = Counter(predicted_values).most_common()
        prediction = most_common_values[0][0]
        return prediction 
    
    def predict(self,X_test:np.array) -> list:
        if X_test.ndim == 1:
            X_test = np.expand_dims(X_test,axis=0)
        
        predictions = np.array([self.__predict(x_pred) for x_pred in X_test])
        return predictions
    
    def accuracy(self,y_true:np.ndarray,y_pred:np.ndarray):
        return np.sum(y_pred == y_true.values) / len(y_pred)