from collections import defaultdict


def M1_algorithm(data, CAR_rules, database_coverage:int = 1, fp_growth:bool = False, include_support:bool = False, verbose:bool = False):
    """
    Create rule-based classifier based on M1 method.
        
    :param data:              Dataset representations
    :param CAR_rules:         Rules generated from CBA-RG stage
    :param database_coverage: Minimal number of rules to cover single data
    :param fp_growth:         Whether the rule generator is based on FP-Growth algorithm
    :param include_support:   Whether to include rule support on the classifier
    :param verbose:           Whether to print intermediate results
    """
    D_size = len(data)
    data_copy = list(enumerate(data))

    if not fp_growth:
        CAR_rules = sorted(CAR_rules, key = lambda x: (x[1][-1]/x[0][-1], x[1][-1], -len(x[0][0])), reverse=True)
    
    classifier = []
    error_history = []
    case_coverage_count = defaultdict(int)

    curr_error = 0

    if verbose:
        print('\nTraining')

    for conds, y in CAR_rules:
        temp = []
        cover_case = False
        r_conds, _ = conds
        r_class, _ = y
        correct_classification = 0

        for id, (d_id, (d_conds, d_class)) in enumerate(data_copy):
            if set(r_conds).issubset(set(d_conds)):
                case_coverage_count[d_id] += 1
                temp.append((id, d_id))

                if r_class == d_class:
                    correct_classification += 1
                    cover_case = True
        
        rule_error = (len(temp) - correct_classification)/D_size
        curr_error += rule_error

        if cover_case:
            if not include_support:
                classifier.append((r_conds, r_class))
            else:
                classifier.append((conds, y))

            for idx, d_id in sorted(temp, key=lambda x: x[0], reverse=True):
                if case_coverage_count[d_id] >= database_coverage:
                    del data_copy[idx]

            default_class, freq = get_default_class(data_copy)
            default_class_error = (len(data_copy) - freq) / D_size

            total_error = curr_error + default_class_error
            error_history.append((total_error, default_class))

            if verbose:
                print('Current training error: ', total_error)
    
    arg_min, min_default_class = min(enumerate(error_history), key=lambda x: x[1][0])
    final_classifier = classifier[0:arg_min+1] + [min_default_class[1]]

    if verbose:
        print('\nNumber of rules in classifier:', len(final_classifier))

    return final_classifier 


def M2_algorithm(data, CAR_rules, verbose:bool = False):
    """
    Create rule-based classifier based on M2 method. This algorithm consists of 3 stages:
       - Stage-1 counts the number of cases covered by each rules
       - Stage-2 decides which rules should become the candidate for classifier
       - Stage-3 creates the final classifier based on previous stages
        
    :param data:      Dataset representation
    :param CAR_rules: Rules generated from CBA-RG stage
    :param verbose:   Whether to print intermediate result
    """
    CAR_rules = [(c, i) for i, c in enumerate(CAR_rules)]

    D_size = len(data)
    Q = set(); U = set(); A = []
    class_case_covered = defaultdict(list)
    mark_rule = defaultdict(lambda: False)
    data_copy = data[:]

    if verbose:
        print('\nTraining')
    
    # Stage-1 of algorithm
    for id, (d_conds, d_class) in enumerate(data_copy):
        correct_class = [c for c in CAR_rules if c[0][1][0][1] == d_class[1]]
        diff_class = [c for c in CAR_rules if c[0][1][0][1] != d_class[1]]
        
        c_rule = max_cover_rule(correct_class, d_conds)
        w_rule = max_cover_rule(diff_class, d_conds)
        
        if c_rule is not None:
            U.add(c_rule)
            class_case_covered[c_rule].append((id, d_class))

            if w_rule is None or higher_precedence(c_rule, w_rule):
                Q.add(c_rule)
                mark_rule[c_rule] = True
            else:
                A.append((id, d_class, c_rule, w_rule))

        else:
            A.append((id, d_class, c_rule, w_rule))
    
    # Stage-2 of algorithm
    replace_rule = defaultdict(set)
    errors_of_rule = defaultdict(int)

    for d_id, d_class, c_rule, w_rule in A:
        if w_rule is not None and mark_rule[w_rule]:
            errors_of_rule[w_rule] += 1
            class_case_covered[w_rule].append((d_id, d_class))

            if c_rule is not None:
                class_case_covered[c_rule].remove((d_id, d_class))
        else:
            w_set = all_cover_rules(U, data[d_id], c_rule)
            
            for w in w_set:
                replace_rule[w].add((c_rule, d_id, d_class))
                class_case_covered[w].append((d_id, d_class))
            
            Q = Q.union(w_set)
    Q = list(Q)

    # Stage-3 of the algorithm
    class_distribution = compute_class_distribution(data)
    id_covered = defaultdict(lambda: False)
    rule_errors = 0
    CBA_classifier = []

    Q = sorted(Q, key = lambda x: (x[1][-1]/x[0][-1], x[1][-1], -x[2]), reverse=True)
    for r in Q:

        if len(class_case_covered[r]) != 0:
            for entry in replace_rule[r]:
                rule, d_id, d_class = entry
                
                if id_covered[d_id]:
                    class_case_covered[r].remove((d_id, d_class))
                else:
                    errors_of_rule[r] += 1
                    if rule is not None:
                        class_case_covered[rule].remove((d_id, d_class))
            
        rule_errors += errors_of_rule[r]

        # update class distribution
        for id, y_class in class_case_covered[r]:
            if not id_covered[id]:
                id_covered[id] = True
                class_distribution[y_class] -= 1

        max_key = max(class_distribution, key = class_distribution.get)
        default_class = ('default_class', max_key[1])
        
        default_errors = sum([v for k, v in class_distribution.items() if k != max_key])
        total_errors = (rule_errors + default_errors) / D_size

        if verbose:
            print('Current training error: ', total_errors)
        
        CBA_classifier.append((r, default_class, total_errors))
    
    arg_min, _ = min(enumerate(CBA_classifier), key=lambda x: x[1][2])
    final_classifier = []

    for i in range(arg_min+1):
        final_classifier.append((CBA_classifier[i][0][0][0], CBA_classifier[i][0][1][0]))

    final_classifier.append(CBA_classifier[i][1])

    if verbose:
        print('\nNumber of rules in classifier:', len(final_classifier))
    
    return final_classifier


def higher_precedence(rule1, rule2):
    """
    Compares two rules and return True if rule1 has higher precedence than rule2.
    A rule1 has higher precedence than rule2 if:
       - conf(rule1) > conf(rule2)
       - if conf(rule1) is equal to conf(rule2) and support(rule1) >= support(rule2)

    :param rule1: Rule that generated from rule generation stage
    :param rule2: Rule that generated from rule generation stage  
    """
    cond1, y1, index1 = rule1
    _, conds_sup_count1 = cond1
    _, rule_sup_count1 = y1
    conf1 = rule_sup_count1/conds_sup_count1

    cond2, y2, index2 = rule2
    _, conds_sup_count2 = cond2
    _, rule_sup_count2 = y2
    conf2 = rule_sup_count2/conds_sup_count2

    # rule 1 has higher precedence if confidence is greater than rule 2's confidence
    if conf1 != conf2:
        return conf1 > conf2

    # rule 1 has higher precedence if rule_sup_count is greater than rule 2's rule_sup_count
    if rule_sup_count1 != rule_sup_count2:
        return rule_sup_count1 > rule_sup_count2

    # tiebreaker by index (which one generated earlier)
    return index1 < index2


def all_cover_rules(U:set, data_row, c_rule):
    """
    Finds all rules that wrongly classify case.d_id and have higher precedence than c_rule.

    :param U:        Set of all rules that covered at least one case
    :param data_row: Single row of dataset
    :param c_rule:   Rule that is correcly classified   
    """
    w_set = set()

    if c_rule is not None:
        c_conf = c_rule[1][1]/c_rule[0][1]
        c_supp = c_rule[1][1]
        try:
            c_rule_idx = c_rule[2]
        except ValueError:
            c_rule_idx = 0
    else:
        c_conf = 0
        c_supp = 0
        c_rule_idx = float("inf")       # If rule is none, rule will not be generated

    for rule in U:
        if rule != c_rule:
            cond, y, index = rule
            r_cond, conds_sup_count = cond
            _, rule_sup_count = y
            conf = rule_sup_count/conds_sup_count
            if set(r_cond).issubset(set(data_row[0])):

                if (conf > c_conf) or (conf == c_conf and (rule_sup_count > c_supp or (rule_sup_count == c_supp and index < c_rule_idx))):
                    w_set.add(rule)
    
    return w_set
            

def max_cover_rule(rules, target_conds):
    """
    Finds the highest precedence rules that cover the input case.

    :param rules:        List of itemrules
    :param target_conds: The conditions (features) of a single row in dataset representation
    """
    cover_rule = None
    temp_conf = 0; temp_supp = 0

    for conds, index in rules:
        r_conds, conds_sup_count = conds[0]
        _, rule_sup_count = conds[1]
        conf = rule_sup_count / conds_sup_count

        if set(r_conds).issubset(set(target_conds)):
            if conf > temp_conf:
                cover_rule = (conds[0], conds[1], index)
                temp_conf = conf
                temp_supp = rule_sup_count
            elif conf == temp_conf and rule_sup_count > temp_supp:
                cover_rule = (conds[0], conds[1], index)
                temp_conf = conf
                temp_supp = rule_sup_count

    return cover_rule


def compute_class_distribution(data, with_d_id = False):
    """
    Calculates the frequency of each class in the dataset

    :param data:      Dataset representation
    :param with_d_id: Whether dataset have unique id
    """
    class_distribution = defaultdict(int)

    if with_d_id:
        for _, (_, y_class) in data:
            class_distribution[y_class] += 1
    else:
        for _, y_class in data:
            class_distribution[y_class] += 1
    
    return class_distribution


def get_default_class(data):
    """
    Get a class with the highest frequency in the remaining dataset

    :param data: Dataset representation
    """
    if len(data) == 0:
        return ('default_class', None), 0

    class_distribution = compute_class_distribution(data, with_d_id = True)
    max_key = max(class_distribution, key = class_distribution.get)
    
    default_class = ('default_class', max_key[1])

    return default_class, class_distribution[max_key]