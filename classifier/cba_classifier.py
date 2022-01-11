from sklearn.base import BaseEstimator, ClassifierMixin

from classifier.rule_generator import apriori_rule_gen
from classifier.classifier_builder import M1_algorithm, M2_algorithm


class CBA(BaseEstimator, ClassifierMixin):
    """
    Classifer based on the class association rule (CAR). This class consists of two stages:
       1. CBA-RG (rule generation) stage, which generate CARs from the dataset using Apriori algorithm
       2. CBA-CB (classifier builder) stage, which construct a classifer from the rules generated in CBA-RG stage
    """
    CONST_CLASS = "class"

    def __init__(self, feature_names = None, support_threshold:float = 0.02, conf_threshold:float = 0.5, method:str = 'M1', max_candidates:int = 80000):
        """
        CBA classifier initialization.
        
        :param feature_names:     Column names of features in dataset
        :param support_threshold: Minimum support (percentage of the dataset)
        :param conf_threshold:    Minimum confidence
        :param method:            Algorithm for CBA-CB stage
        :param max_candidates:    Maximum number of candidates generated in CBA-RG stage
        """
        if method not in ["M1", "M2"]:
            raise ValueError("Method must be M1 or M2!")

        self.feature_names = feature_names
        self.support_threshold = support_threshold
        self.conf_threshold = conf_threshold
        self.method = method
        self.max_candidates = max_candidates

        self.rule_generator = apriori_rule_gen
        self.classifier = []

        if self.method == "M1":
            self.classifier_builder = M1_algorithm
        elif self.method == "M2":
            self.classifier_builder = M2_algorithm

    def _preprocess_train(self, X, y):
        """
        Pre-process the training dataset into trainable representation.

        :param X: Dataset features
        :param y: Dataset ground truth class
        """
        dataset_x, dataset_y = [], []

        for i in range(len(X)):
            current_row = []

            for feature_index, feature_name in enumerate(self.feature_names):
                current_row.append((feature_name, X[i][feature_index]))

            dataset_x.append(current_row)
            dataset_y.append((self.CONST_CLASS, y[i]))

        return list(zip(dataset_x, dataset_y))

    def _preprocess_test(self, X):
        """
        Pre-process the test dataset (does not contain label).

        :param X: Dataset features
        """
        dataset_x = []

        for i in range(len(X)):
            current_row = []

            for feature_index, feature_name in enumerate(self.feature_names):
                current_row.append((feature_name, X[i][feature_index]))
            
            dataset_x.append(current_row)

        return dataset_x

    def fit(self, X, y, verbose:bool = False):
        """
        Train the model based on the input dataset.
        
        :param X:       2D-arrays of dataset features
        :param y:       Dataset target class
        :param verbose: Whether to print intermediate results
        """
        if self.feature_names is None:
            self.feature_names = [f'features_{i}' for i in range(len(X[0]))]

        transactions = self._preprocess_train(X, y)
        data_size = len(X)

        min_support = self.support_threshold * data_size
        CAR_rules = self.rule_generator(transactions, min_support = min_support, min_conf = self.conf_threshold, max_candidates = self.max_candidates, verbose = verbose)
        
        if verbose:
            print("Total CARs generated:", len(CAR_rules))

        if len(CAR_rules) == 0:
            raise ValueError("The model does not generate any rules. Try to lower support_threshold or conf_threshold")

        self.classifier = self.classifier_builder(transactions, CAR_rules, verbose=verbose)

        if len(self.classifier) == 0:
            raise RuntimeWarning("The model does not learn anything. Try to change support_threshold or conf_threshold")

        return self

    def predict(self, X_test):
        """
        Make predictions based on test dataset.
        
        :param X_test: Test dataset features
        """
        assert len(self.classifier) > 0, "Classifier have not been trained!"

        X_test = self._preprocess_test(X_test)

        predictions = []
        for d in X_test:
            for rule in self.classifier:
                if rule[0] != 'default_class' and set(rule[0]).issubset(set(d)):
                    predictions.append(rule[1][1])
                    break
                elif rule[0] == 'default_class':
                    predictions.append(rule[1])

        return predictions