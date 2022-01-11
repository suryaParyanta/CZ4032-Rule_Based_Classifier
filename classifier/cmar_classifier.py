from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin

from classifier.rule_generator import fp_growth_rule_generation, chi_square_test
from classifier.classifier_builder import M1_algorithm


class CMAR(BaseEstimator, ClassifierMixin):
    """
    Classifer based on the class association rule (CAR). This class consists of two stages:
       1. CMAR-RG (rule generation) stage, which generate CARs from the dataset using FP-Growth algorithm
       2. CMAR-CB (classifier builder) stage, using M1 algorithm to construct a classifer from the rules generated in CMAR-RG stage
    """

    CONST_CLASS = "class"

    def __init__(self, features_name:list = None, support_threshold:float = 0.02, conf_threshold:float = 0.5, database_coverage:int = 4):
        """
        CBA classifier initialization.
        
        :param features_name:     Column names of features in dataset
        :param support_threshold: Minimum support (percentage of the dataset)
        :param conf_threshold:    Minimum confidence
        :param database_coverage: Minimum rules covered for each case (row) in dataset
        """
        self.features_name = features_name
        self.support_threshold = support_threshold
        self.conf_threshold = conf_threshold
        self.database_coverage = database_coverage

        self.rule_generator = fp_growth_rule_generation
        self.classifier_builder = M1_algorithm

        self.dataset_size = None
        self.dataset_dist = None
        self.classifier = []

    def _preprocess_train(self, X, y):
        """
        Pre-process the training dataset into trainable representation.

        :param X: Dataset features
        :param y: Dataset ground truth class
        """
        dataset_x, dataset_y = [], []

        for i in range(len(X)):
            current_row = []
            for feature_index, feature_name in enumerate(self.features_name):
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
            for feature_index, feature_name in enumerate(self.features_name):
                current_row.append((feature_name, X[i][feature_index]))
            dataset_x.append(current_row)

        return dataset_x

    def fit(self, X, y, verbose: bool = False):
        """
        Train the model based on the input dataset.
        
        :param X:       2D-arrays of dataset features
        :param y:       Dataset target class
        :param verbose: Whether to print intermediate results
        """
        if self.features_name is None:
            self.features_name = [f"features_{i}" for i in range(len(X[0]))]

        transactions = self._preprocess_train(X, y)
        self.dataset_size = len(X)

        min_support = self.support_threshold * self.dataset_size
        CAR_rules, dataset_dist = self.rule_generator(transactions, min_support = min_support, min_conf = self.conf_threshold, verbose = verbose)
        self.dataset_dist = dataset_dist
        
        if verbose:
            print("Total CARs generated:", len(CAR_rules))
            
        if len(CAR_rules) == 0:
            raise ValueError("The model does not generate any rules. Try to lower support_threshold or conf_threshold")

        self.classifier = self.classifier_builder(transactions, CAR_rules, database_coverage = self.database_coverage, fp_growth = True, include_support = True, verbose=verbose)
        
        if len(self.classifier) == 0:
            raise RuntimeWarning("The model does not learn anything. Try to change support_threshold or conf_threshold")

        return self

    def predict(self, X_test):
        """
        Make predictions based on test dataset.
        
        :param X_test: Test dataset features
        """
        assert len(self.classifier) > 0, "CMAR classifier have not learned anything yet!"

        X_test = self._preprocess_test(X_test)

        predictions = []
        for d in X_test:
            rules_covered = defaultdict(list)
            go_to_default = True
            
            for rule in self.classifier:
                if rule[0] != "default_class":
                    conds, y = rule
                    conditions, _ = conds
                    y_class, _ = y

                    if set(conditions).issubset(set(d)):
                        go_to_default = False
                        rules_covered[y_class].append(rule)

                elif rule[0] == "default_class" and go_to_default:
                    predictions.append(rule[1])
                
            if go_to_default:
                continue
            
            max_weighted_chi2 = 0
            y_predict = None
            for y_class, rules in rules_covered.items():
                weighted_chi2 = 0

                for rule in rules:
                    conds, _ = rule
                    _, cond_sup_count = conds
                    
                    _, chi2_value = chi_square_test(rule, self.dataset_dist)

                    e = 1 / (cond_sup_count * self.dataset_dist[y_class]) + \
                        1 / (cond_sup_count * (self.dataset_size - self.dataset_dist[y_class])) + \
                        1 / ((self.dataset_size - cond_sup_count) * self.dataset_dist[y_class]) + \
                        1 / ((self.dataset_size - cond_sup_count) * (self.dataset_size - self.dataset_dist[y_class]))

                    max_chi_2 = self.dataset_size * e * (min(cond_sup_count, self.dataset_dist[y_class]) - cond_sup_count * self.dataset_dist[y_class] / self.dataset_size)**2
                    weighted_chi2 += chi2_value**2 / max_chi_2
                
                if weighted_chi2 > max_weighted_chi2:
                    max_weighted_chi2 = weighted_chi2
                    y_predict = y_class
            
            predictions.append(y_predict[1])
                
        return predictions