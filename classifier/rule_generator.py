from collections import defaultdict
from itertools import combinations


################################################### Rule Generation using A-priori ####################################################

def apriori_rule_gen(data:list, 
                     min_support:float = 3.0, min_conf:float = 0.5, max_candidates:int = 80000, 
                     verbose:bool = False):
    """
    Generates Class Association Rules (CARs) with Apriori-based algorithm. The algorithm is as follows:
       1. Generate set of frequent 1-ruleitems (called F-1)
       2. Get the actual rules 1-CAR by filtering out F-1 with minimum confidence threshold
       3. Generate candidate of k-ruleitems (called C-k) by using F-1 and F-(k-1)
       4. Get the set of frequent k-ruleitems (F-k) by filtering out C-k with minimum support threshold
       5. Get the actual rules k-CAR by filtering out F-k with minimum confidene threshold   
       6. Repeat step 3-5 with k = 2,3,4,... until the number of rules in k-CAR is equal to one

    :param data:           Training list of data. It is in the form of:
                              [((condition1, value1), (condition2, value2), ..., (condition_x, value_x), (ground_truth, class_1))
                                (condition1, value1), (condition2, value2), ..., (condition_x, value_x), (ground_truth, class_2),
                                ...
                              )]
    :param min_support:    Minimum threshold of rule support count conditions
    :param min_conf:       Minimum threshold of rule confidence
    :param max_candidates: Maximum total number of candidates generated
    :param verbose:        Whether to print intermediate results
    """
    class_assoc_rules = []

    # First iteration: generate F1 and CAR1
    support_counter = defaultdict(lambda: defaultdict(int))
    for conditions, label in data:
        for item in conditions:
            support_counter[item]['condSupCount'] += 1
            support_counter[item][label] += 1
        
    F1 = [(((conditions,), support_counter[conditions]['condSupCount']), (key, val)) \
          for conditions in support_counter.keys() for key, val in support_counter[conditions].items() if key != 'condSupCount' and val > min_support]
    Fk = F1

    CAR_1 = gen_rules(F1, support_counter, min_conf=min_conf)
    class_assoc_rules.append(CAR_1)

    if verbose:
        print('Iteration 1')
        print('Number of 1-ruleitem generated:', len(CAR_1))

    # Next iterations: generate Fk and CARk, where k >= 2
    iteration = 2
    remaining_candidates = max_candidates
    while len(Fk) > 0 and remaining_candidates > 0:
        next_candidates = generate_candidates(F1, Fk, remaining_candidates)
        remaining_candidates -= len(next_candidates)

        support_counter = defaultdict(lambda: defaultdict(int))

        for X, y in data:
            supported_candidates = rule_subset(next_candidates, X)
            
            for conditions, label in supported_candidates:
                support_counter[conditions]['condSupCount'] += 1
                if label == y:
                    support_counter[conditions][y] += 1
        
        Fk = [((conditions, support_counter[conditions]['condSupCount']), (key, val)) \
              for conditions in support_counter.keys() for key, val in support_counter[conditions].items() if key != 'condSupCount' and val > min_support]

        CAR_k = gen_rules(Fk, support_counter, min_conf = min_conf)
        class_assoc_rules.append(CAR_k)

        if verbose:
            print('\nIteration', iteration)
            print(f'Number of {iteration}-ruleitems generated: {len(CAR_k)}')
        
        iteration += 1
    
    return sum(class_assoc_rules, [])


def gen_rules(k_frequent_rules, support_counter:dict, min_conf:float = 0.5):
    """
    Filters the frequent k-ruleitems based on the minimum confidence.

    :param k_frequent_rules: List of frequent k-ruleitems
    :param support_counter:  Counter that stores support and confidence of all candidate rules
    :param min_conf:         Minimum threshold of rule confidence        
    """
    k_CAR = []
    temp = []

    for condset, y in k_frequent_rules:
        add_to_CAR = True
        y_class, rule_sup_count = y
        conditions, cond_sup_count = condset
        conf = rule_sup_count/cond_sup_count

        if len(conditions) == 1:
            conditions = conditions[0]

        max_rule_sup = max([support_counter[conditions][k] for k in support_counter[conditions].keys() if k != 'condSupCount'])

        if rule_sup_count < max_rule_sup:
            continue
        elif conf <= min_conf:
            continue

        for k in support_counter[conditions].keys():
            if k != 'condSupCount' and y_class != k and rule_sup_count == support_counter[conditions][k]:
                if conditions not in temp:
                    temp.append(conditions)
                else:
                    add_to_CAR = False
        
        if add_to_CAR:
            k_CAR.append((condset, y))
    
    return k_CAR
        

def generate_candidates(F1, Fk, remaining_candidates: int):
    """
    Proposes the candidates for k+1-ruleitems generated from k-ruleitems and 1-ruleitems.

    :param F1:                   Frequent 1-ruleitems
    :param Fk:                   Frequent k-ruleitems
    :param remaining_candidates: Maximum number of candidates that can be generated
    """
    stored_as_candidates = defaultdict(lambda: False)
    candidates = []
    for ruleitem1 in F1:
        for ruleitem2 in Fk:
            if len(candidates) == remaining_candidates:
                break

            condset1, y1 = ruleitem1
            condset2, y2 = ruleitem2

            y1_class, _ = y1
            conditions1, _ = condset1

            y2_class, _ = y2
            conditions2, _ = condset2

            cond1_feature = conditions1[0][0]
            cond2_features = [c[0] for c in conditions2]
            if y1_class == y2_class and cond1_feature not in cond2_features:
                new_cond = tuple(sorted([conditions1[0]] + list(conditions2), key= lambda x: x[0]))
            
                if not stored_as_candidates[new_cond]:
                    candidates.append([new_cond, y1_class])
                    stored_as_candidates[new_cond] = True

    return candidates


def rule_subset(rules, transaction_conds):
    """
    Get all rules that cover the specified transaction conditions (features). 
    A rule is said to cover a transaction conditions if rule conditions are subset of transaction conditions.

    :param rules:             List of ruleitems
    :param transaction_conds: The conditions (features) of a single row in dataset
    """
    x_set = set(transaction_conds)
    supported_candidates = []

    for conds, y in rules:
        if set(conds).issubset(x_set):
            supported_candidates.append([conds, y])
    
    return supported_candidates

#######################################################################################################################################


################################################### Rule Generation using FP-Growth ###################################################

class FPNode:
    """
    FP-tree/CR-tree data structure implementation. One node in FP-tree/CR-tree contains five items:
      1. Child nodes: children of the current node (can be multiple nodes)
      2. Parent node: parent of the current node (only one parent)
      3. Value: pattern (condset) of one ruleitem
      4. Next node: pointer point to another node with same value
      5. Class information: stores information about class frequency or support/confidence of a rule
    """

    def __init__(self, val = None, class_dist = None, next_node = None, parent = None, child = None, cr_tree:bool = False):
        """
        Initialization of FP-tree node.

        :param val:        Pattern (condset) of one ruleitem
        :param class_dist: Counter to count the class frequency
        :param next_node:  Pointer to the next node with same node value
        :param parent:     Parent of current node
        :param child:      The children of current node
        :param cr_tree:    Whether to build CR-tree
        """
        self.val = val
        self.class_dist = class_dist
        self.parent = parent
        self.next_node = next_node
        self.cr_tree = cr_tree

        if class_dist is None and not self.cr_tree:
            self.class_dist = defaultdict(int)
        elif class_dist is None:
            self.class_dist = defaultdict(list)

        if child is None:
            self.child = defaultdict(lambda: None)

    def merge_with_parent(self):
        """
        Combine class distribution of current node with parent node
        """
        parent_node = self.parent

        for k in self.class_dist.keys():
            parent_node.class_dist[k] += self.class_dist[k]

        parent_node.child[self.val] = None


def fp_growth_rule_generation(data, min_support:float = 3.0, min_conf:float = 0.5, verbose:bool = False, **kwargs):
    """
    Generates Class Association Rules (CARs) with FP-Growth algorithm. The algorithm is as follows:
       1. Generate frequent itemset by using minimum support as threshold.
       2. Sort the frequent itemset by its frequency in descending order, the result is called F-order.
       3. Rearrange the dataset features according to the F-order.
       4. Create header table based on F-order and generate the FP-tree.
       5. Generate CARs by mining the FP-tree recursively.
       6. Store the CARs in CR-tree.
    
    :param data:        Training list of data. It is in the form of:
                           [((condition1, value1), (condition2, value2), ..., (condition_x, value_x), (ground_truth, class_1))
                             (condition1, value1), (condition2, value2), ..., (condition_x, value_x), (ground_truth, class_2),
                              ...
                           )]
    :param min_support: Minimum threshold of rule support count conditions
    :param min_conf:    Minimum threshold of rule confidence
    :param verbose:     Whether to print intermediate results
    """
    # Get frequent itemset
    freq_itemset = get_frequent_itemset(data, min_support = min_support)
    freq_itemset = sorted(freq_itemset, key = lambda x: x[1], reverse=True)
    F_list = {key: idx for idx, (key, _) in enumerate(freq_itemset)}
    F_order = sorted(F_list, key=F_list.get)

    # Clean the dataset and get dataset_distribution
    dataset_dist = defaultdict(int)
    data_clean = []
    for conditions, label in data:
        dataset_dist[label] += 1
        common_features = sorted([c for c in conditions if c in F_list], key = lambda x: F_list[x])
        data_clean.append((common_features, label))

    # Mine the dataset
    head_node = FPNode()
    header_table = create_header_table(freq_itemset)

    for conditions, label in data_clean:
        insert_feature(header_table, head_node, conditions, label)

    generated_rules = fp_growth(F_order, header_table, (), dataset_dist, min_support = min_support, min_conf = min_conf)

    if verbose:
        print('Number of generated rules using FP-Growth:', len(generated_rules))

    # Store rules in CR-Tree and perform pruning
    generated_rules = sorted(generated_rules, key = lambda x: (x[1][-1]/x[0][-1], x[1][-1], -len(x[0][0])), reverse=True)
    
    CR_tree = FPNode(cr_tree = True)
    CR_header_table = create_header_table(freq_itemset)
    rule_feature_count = defaultdict(int)

    for conds, _ in generated_rules:
        conditions, _ = conds
        
        for c in conditions:
            rule_feature_count[c] += 1
    
    feature_order = sorted(rule_feature_count.keys(), key = lambda x: rule_feature_count[x], reverse=True)
    feature_idx = {k:idx for idx, k in enumerate(feature_order)}
    prune_idx = []

    for idx, (conds, y) in enumerate(generated_rules):
        conditions, cond_sup_count = conds
        y_class, rule_sup_count = y
        conf = rule_sup_count/cond_sup_count

        common_features = sorted([c for c in conditions if c in feature_idx], key = lambda x: feature_idx[x])

        if CR_header_table[common_features[-1]]['head'] is None:
            insert_feature(CR_header_table, CR_tree, common_features, y_class, cr_tree = True, cond_support = cond_sup_count, rule_support = rule_sup_count)
        else:
            suffix_feature = common_features.pop()
            curr_node = CR_header_table[suffix_feature]['head']
            prune_rule = False

            while curr_node is not None:
                tree_path, class_dist = traverse_FP_tree(curr_node)

                if len(class_dist) != 0 and set(tree_path).issubset(set(common_features)):
                    for v in class_dist.values():
                        subset_conf = v[0]/v[1]

                        if subset_conf > conf:
                            prune_rule = True
                            break

                        elif subset_conf == conf and v[0] > rule_sup_count:
                            prune_rule = True
                            break

                        elif subset_conf == conf and v[0] == rule_sup_count and len(tree_path) < len(common_features):
                            prune_rule = True
                            break

                if prune_rule:
                    prune_idx.append(idx)
                    break

                curr_node = curr_node.next_node
    
    while len(prune_idx) > 0:
        idx = prune_idx.pop()
        del generated_rules[idx]

    if verbose:
        print('Number of rules after pruning:', len(generated_rules))
    
    return generated_rules, dataset_dist


def fp_growth(F_order, header_table, curr_pattern, dataset_dist, min_support:float = 3.0, min_conf:float = 0.5):
    """
    Recursively mine the conditional database. This algorithm is based on pseudocode given in the 
    original paper (https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf). The algorithm is as follows:
       For itemset in F_order.reverse():
          1. Get the corresponding nodes in the FP-tree (the nodes should be at bottom level).
          2. For each node n: 
                - Traverse to the upper level until reached root node
                - While traversing, merge n with its parent node
             Store all of the traversed path into one array, called paths.
          3. If number of paths is only one, generate the rules by looking at all combinations. Prune out
             some rules if they do not pass chi-square test.
          4. If number of paths is greater than one, then:
                - Generate the n-projected database by finding new frequent itemsets and F-order
                - Generate a new FP-Tree based on the n-projected database (called this function again) 

    :param F_order:      List of frequent itemset sorted from their frequency
    :param header_table: Header table of current FP-Tree
    :param curr_pattern: Pattern suffix
    :param dataset_dist: Frequency of each class in the dataset         
    :param min_support:  Minimum threshold of rule support count conditions
    :param min_conf:     Minimum threshold of rule confidence
    """
    generated_rules = []
    
    for _ in range(len(F_order)):
        itemset = F_order.pop()
        paths = []
        start_node = header_table[itemset]['head']
        curr_node = start_node
        curr_node_dist = defaultdict(int)

        while curr_node is not None:
            tree_path, class_dist = traverse_FP_tree(curr_node)

            for k in class_dist:
                curr_node_dist[k] += class_dist[k]

            paths.append((tree_path, class_dist))
            curr_node.merge_with_parent()
            curr_node = curr_node.next_node
        
        pattern_rule = (start_node.val,) + curr_pattern
        conds_sup_count = sum(curr_node_dist.values())
        max_key = max(curr_node_dist, key=curr_node_dist.get)

        assert conds_sup_count == header_table[itemset]['frequency'], f'{itemset} frequency is not match!'

        conf = curr_node_dist[max_key]/conds_sup_count
        if curr_node_dist[max_key] > min_support and  conf > min_conf:
            rule = ((pattern_rule, conds_sup_count), (max_key, curr_node_dist[max_key]))
            valid, _ = chi_square_test(rule, dataset_dist)

            # if a rule is success on chi squeare test, add to CR-tree 
            if valid:
                generated_rules.append(rule)

        # if confidence is equal to one or the support is less than threshold no need to create projected database 
        if conf == 1 or curr_node_dist[max_key] <= min_support:
            continue

        # if tree-traversal only generate 1 path, generate all possible combination
        if len(paths) == 1:
            rules = []
            tree_path, class_dist = paths[0]
            conds_sup_count = sum(class_dist.values())
            max_key = max(class_dist, key = class_dist.get)

            for i in range(1, len(tree_path)+1):
                for pattern in combinations(tree_path, i):
                    new_pattern = pattern + pattern_rule

                    if class_dist[max_key] > min_support and class_dist[max_key]/conds_sup_count > min_conf:
                        rule = ((new_pattern, conds_sup_count), (max_key, class_dist[max_key]))
                        valid, _ = chi_square_test(rule, dataset_dist)

                        if valid:
                            rules.append(rule)
            
            generated_rules += rules

        # if more than 1 path generated, create projected (conditional) database and recursively call the function
        elif len(paths) > 1:
            projected_database = create_projected_database(paths)

            new_freq_itemset = get_frequent_itemset(projected_database, min_support = min_support, with_freq = True)
            new_freq_itemset = sorted(new_freq_itemset, key = lambda x: x[1], reverse=True)

            if len(new_freq_itemset) == 0:
                continue

            F_list = {key: idx for idx, (key, _) in enumerate(new_freq_itemset)}
            new_F_order = sorted(F_list, key = F_list.get)

            new_header_table = create_header_table(new_freq_itemset)
            conditional_fp_tree = FPNode()
            clean_database = []

            for (conditions, label), freq in projected_database:
                common_features = sorted([c for c in conditions if c in F_list], key = lambda x: F_list[x])
                clean_database.append(((common_features, label), freq))
            
            for (conditions, label), freq in clean_database:
                insert_feature(new_header_table, conditional_fp_tree, conditions, label, value = freq)

            generated_rules += fp_growth(new_F_order, new_header_table, pattern_rule, dataset_dist, min_support=min_support, min_conf=min_conf)

    return generated_rules


def chi_square_test(rule, dataset_dist, significance_level:float = 0.05):
    """
    Perform Chi Square test.

    :param rule:               Ruleitem
    :param dataset_dist:       Frequency of each class in the dataset 
    :param significance_level: Significance level in Degree of Freedom table 
    """
    DF_1_TABLE = {
        0.06:  3.5374,
        0.05:  3.8415,
        0.04:  4.2179,
        0.03:  4.7093,
        0.025: 5.0239,
        0.02:  5.4119
    }

    conds, y = rule
    _, conds_sup_count = conds
    y_class, rule_sup_count = y

    y_true = dataset_dist[y_class]
    y_false = sum([v for k, v in dataset_dist.items() if k != y_class])
    dataset_size = y_true + y_false

    # create confusion matrix and expected matrix
    confusion_matrix = [[None, None],[None, None]]
    confusion_matrix[0][0] = rule_sup_count
    confusion_matrix[0][1] = conds_sup_count - rule_sup_count
    confusion_matrix[1][0] = y_true - rule_sup_count
    confusion_matrix[1][1] = y_false - confusion_matrix[0][1]

    expected_matrix = [[None, None], [None, None]]
    expected_matrix[0][0] = conds_sup_count * y_true / dataset_size
    expected_matrix[0][1] = conds_sup_count * y_false / dataset_size
    expected_matrix[1][0] = (dataset_size - conds_sup_count) * y_true / dataset_size
    expected_matrix[1][1] = (dataset_size - conds_sup_count) * y_false / dataset_size

    chi2_value = 0
    for i in range(2):
        for j in range(2):
            if expected_matrix[i][j] == 0:
                chi2_value += float('inf')
            else:
                chi2_value += (confusion_matrix[i][j] - expected_matrix[i][j])**2 / expected_matrix[i][j]
    
    return chi2_value > DF_1_TABLE[significance_level], chi2_value


def create_projected_database(tree_paths):
    """
    Create conditional database from set of rules.

    :param tree_paths: Paths from FP-tree traversal 
    """
    projected_database = []

    for path, class_dist in tree_paths:
        for k in class_dist:
            projected_database.append(((path, k), class_dist[k]))
    
    return projected_database


def traverse_FP_tree(node:FPNode):
    """
    FP-tree traversal from bottom node.

    :param node: Starting node 
    """
    curr_node = node
    class_dist = curr_node.class_dist
    path = []

    while curr_node is not None:
        if curr_node.val is not None:
            path.append(curr_node.val)
        curr_node = curr_node.parent
    
    path = path[::-1]
    path.pop()

    return path, class_dist


def insert_feature(header_table, head_node, features, label, value:int = 1, cr_tree:bool = False, **kwargs):
    """
    Add the features into FP-Tree or CR-tree.

    :param header_table: Header table of FP-tree/CR-tree
    :param head_node:    Root node of FP-tree/CR-tree 
    :param features:     Pattern of a rule 
    :param label:        Class information of a rule 
    :param value:        Frequency of a rule
    :param cr_tree:      Whether using CR-tree
    :param kwargs:       Store additional information (support/confidence) to CR-tree 
    """
    curr_node = head_node

    for idx, f in enumerate(features):
        if curr_node.child[f] is None:
            new_node = FPNode(val=f, parent=curr_node, cr_tree=cr_tree)
            curr_node.child[f] = new_node

            # update header table
            if header_table[f]['tail'] is None:
                header_table[f]['head'] = new_node
                header_table[f]['tail'] = new_node
            else:
                header_table[f]['tail'].next_node = new_node
                header_table[f]['tail'] = header_table[f]['tail'].next_node
        
        # go to child node
        curr_node = curr_node.child[f]

        # update class distribution on the last node
        if idx == len(features) - 1 and not cr_tree:
            curr_node.class_dist[label] += value
        elif idx == len(features) - 1:
            curr_node.class_dist[label] += [kwargs['rule_support'], kwargs['cond_support']]


def get_frequent_itemset(data, min_support:float = 3.0, with_freq:bool = False):
    """
    Generate frequent itemset from (conditional) database.

    :param data:        Dataset/database
    :param min_support: Minimum threshold of rule support count conditions
    :param with_freq:   Whether dataset include frequency of each item
    """
    freq_counter = defaultdict(int)

    if not with_freq:
        for conditions, _ in data:
            for c in conditions:
                freq_counter[c] += 1
    else:
        for (conditions, _), freq in data:
            for c in conditions:
                freq_counter[c] += freq

    freq_itemset = [(k, v) for k, v in freq_counter.items() if v > min_support]

    return freq_itemset


def create_header_table(freq_itemset):
    """
    Create header table that consists of:
       - Column 1: feature
       - Column 2: frequency
       - Column 3: head node of queue
       - Column 4: tail node of queue

    :param freq_itemset: Frequent itemsets
    """
    header_table = defaultdict(dict)

    for item, freq in freq_itemset:
        header_table[item]['frequency'] = freq
        header_table[item]['head'] = None
        header_table[item]['tail'] = None

    return header_table