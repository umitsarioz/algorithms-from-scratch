import numpy as np 

class Node:
    def __init__(self,feature_index=None,threshold=None,condition_mark=None,left=None,right=None,score=None,criterion=None,information_gain=None,label=None):
        self.feature_index = feature_index
        self.threshold = threshold 
        self.condition_mark = condition_mark 
        self.left = left 
        self.right = right 
        self.score = score 
        self.criterion = criterion
        self.information_gain = information_gain
        self.label = label 
        
class DecisionTree:
    def __init__(self,max_depth=5,min_samples_count=3,criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_count = min_samples_count 
        self.tree = None 
        self.criterion = criterion 
        self.depth = 0 
    
    def _should_stop(self,data,depth):
        n_labels = len(np.unique(data[:,-1]))
        n_samples = len(data)
        condition = (n_labels == 1) or (n_samples <= self.min_samples_count) or (depth >= self.max_depth)
        return condition 
    
    def _get_label_as_majority(self,data):
        labels,counts = np.unique(data[:,-1],return_counts=True)
        idx_max = np.argmax(counts)
        return labels[idx_max]
    
    def _get_potential_splits(self,data):
        potential_splits_all = []
        n_features = data.shape[1] - 1 # [feat_1,feat_2,...,feat_n,labels]
        for feature_idx in range(n_features): # iterate over all features
            data_feature = data[:,feature_idx] 
    
            if isinstance(data_feature[0],str) or isinstance(data_feature[0],bool):
                thresholds = np.unique(data_feature)
                condition_mark = '=='
                potential_splits_all.append({'idx':feature_idx,'thresholds':thresholds,'condition_mark':condition_mark})
            else:
                thresholds = [np.median(data_feature)]
                condition_mark = '<='
                potential_splits_all.append({'idx':feature_idx,'thresholds':thresholds,'condition_mark':condition_mark})
        return potential_splits_all
    
    def _find_best_split(self,data,potential_splits):
        besties = {'feature_index':None,'threshold':None,'condition_mark':None,
                   'information_gain':-float("inf"),'impurity':None,'left_idxs':None,'right_idxs':None}
        
        labels = data[:,-1]
        
        for row in potential_splits:
            feature_idx = row["idx"]
            thresholds = row["thresholds"]
            condition_mark = row["condition_mark"]
            features = data[:,feature_idx]

            for threshold in thresholds:
                if condition_mark == '==': # for categorical features 
                    cond = np.array([x == threshold for x in features])
                else: # for numerical features 
                    cond = np.array([x <= threshold for x in features])
                    
                left_idxs = np.where(cond)[0]
                right_idxs = np.where(~cond)[0]
                information_gain,impurity = self._calculate_information_gain(labels, left_idxs, right_idxs)  
                if information_gain > besties['information_gain']:
                    dct = {'feature_index':feature_idx,'threshold':threshold,
                           'condition_mark':condition_mark,'information_gain':information_gain,'impurity':impurity,
                           'left_idxs':left_idxs,'right_idxs':right_idxs}
                    besties.update(dct)
        
        return besties 
                   
                    
    def _calculate_information_gain(self,labels, left_idxs, right_idxs):
        if len(left_idxs) == 0 or len(right_idxs) == 0 :
            information_gain, weighted_impurity = 0 ,0 
            return information_gain, weighted_impurity 
        else:
            p_left = len(left_idxs) / len(labels)
            p_right = 1 - p_left 

            weighted_impurity = p_left * self._calculate_impurity(labels[left_idxs]) + p_right * self._calculate_impurity(labels[right_idxs])
            parent_impurity = self._calculate_impurity(labels)

            information_gain = parent_impurity - weighted_impurity
            return information_gain, weighted_impurity
    
    def _calculate_impurity(self,labels):
        if self.criterion == 'gini':
            return self._calculate_gini(labels)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(labels)
        else:
            raise Exception("Criterion must be 'gini' or 'entropy'.")
            
    def _calculate_entropy(self,labels):
        _,counts= np.unique(labels,return_counts=True)
        probs = counts / np.sum(counts)
        score = -np.sum(probs*np.log2(probs+1e-9))# Add small value to avoid log(0)
        return score 
    
    def _calculate_gini(self,labels):
        _,counts= np.unique(labels,return_counts=True)
        probs = counts / np.sum(counts)
        score = 1 - np.sum(np.power(probs,2)) 
        return score 
    
    def _build_tree(self,data,depth=0):
        if self._should_stop(data,depth):
            leaf_label = self._get_label_as_majority(data)
            return Node(label=leaf_label)
        else:
            potential_splits = self._get_potential_splits(data)
            besties = self._find_best_split(data,potential_splits)

            left_tree = self._build_tree(data = data[besties['left_idxs']],depth=depth+1)
            right_tree = self._build_tree(data = data[besties['right_idxs']],depth = depth+1)
            
            return Node(feature_index=besties['feature_index'],threshold=besties['threshold'],
                        condition_mark=besties['condition_mark'],left=left_tree,right=right_tree,
                        score=besties['impurity'],criterion=self.criterion,
                        information_gain=besties['information_gain'])
        
    def fit(self,X,y):
        data = np.column_stack((X,y))
        self.tree = self._build_tree(data) 
    
    def predict(self,X):
        predictions = [self._traverse_tree(data_point, self.tree) for data_point in X]
        return predictions

    # Traverse the tree recursively to predict the label
    def _traverse_tree(self, data_point, node):
        if node.label is not None:  # If we're at a leaf, return the label
            return node.label
        
        if node.condition_mark == '==':
            if data_point[node.feature_index] == node.threshold:
                return self._traverse_tree(data_point, node.left)
            else:
                return self._traverse_tree(data_point, node.right)
        else:
            if data_point[node.feature_index] <= node.threshold:
                return self._traverse_tree(data_point, node.left)
            else:
                return self._traverse_tree(data_point, node.right)
            
            
def test_with_iris_dataset():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    data = load_iris()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTree(max_depth=3, min_samples_count=2, criterion='gini')
    tree.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = tree.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gini Test accuracy: {accuracy * 100:.2f}%")

    # Create and train the decision tree with 'entropy' criterion
    tree_entropy = DecisionTree(max_depth=3, min_samples_count=2, criterion='entropy')
    tree_entropy.fit(X_train, y_train)

    # Make predictions on the test set using entropy criterion
    y_pred_entropy = tree_entropy.predict(X_test)

    # Calculate accuracy
    accuracy_entropy =  accuracy_score(y_test, y_pred_entropy)
    print(f"Entropy Test accuracy: {accuracy_entropy * 100:.2f}%")


def test_with_vine_dataset():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_wine()
    X, y = data.data, data.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and fit the custom decision tree
    tree = DecisionTree(max_depth=5, min_samples_count=3, criterion='entropy')
    tree.fit(X_train, y_train)

    # Predict on the test set
    y_pred = tree.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ =="__main__":
    print("running iris dataset:")
    test_with_iris_dataset()
    print("runnig test vine")
    test_with_vine_dataset()
