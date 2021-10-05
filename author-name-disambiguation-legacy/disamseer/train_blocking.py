import copy
import math
import operator
import os
import pickle
import sys
import time

import pandas as pd

import generate_label_for_blocking


def calc_stat(pairs, learned_blocking, type):
    RR = 0.0
    PC = 0.0
    F = 0.0

    if type == 'CNF':
        Pos = pairs['match'] == True
        result = pd.Series([True] * len(pairs))

        for term in learned_blocking:
            L = pd.Series([False] * len(pairs))
            for predicate in term:
                L |= pairs[predicate] == True
            result &= L

        RR = 1.0 - len(pairs[result]) / float(len(pairs))
        PC = len(pairs[Pos & result]) / float(len(pairs[Pos]))
        F = 2 * RR * PC / (RR + PC)

    else:
        Pos = pairs['match'] == True
        result = pd.Series([False] * len(pairs))

        for term in learned_blocking:
            L = pd.Series([True] * len(pairs))
            for predicate in term:
                L &= (pairs[predicate] == True)
            result |= L

        RR = 1.0 - len(pairs[result]) / float(len(pairs))
        PC = len(pairs[Pos & result]) / float(len(pairs[Pos]))
        F = 2 * RR * PC / (RR + PC)

    print "RR : " + str(RR)
    print "PC : " + str(PC)
    print "F  : " + str(F)

def rank_keys(label_file):
    def DNF_gain(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        #print P
        #print N
        #print p
        #print n
        if p == 0:
            return 0
        else:
            return p * (math.log(p/(p+n)) - math.log(P/(P+N)))

    labeled_pairs = pd.read_pickle(label_file)
    cols_to_use = generate_label_for_blocking.COLS_TO_USE_APPENDED
    gain_dict = dict()

    for key in cols_to_use:
        L = labeled_pairs[key] == True
        pos_idx = labeled_pairs['match'] == True
        neg_idx = ~pos_idx
        gain = DNF_gain(labeled_pairs, L, pos_idx, neg_idx)
        gain_dict[key] = gain

    sorted_gain_dict = sorted(gain_dict.items(), key=operator.itemgetter(1))
    print sorted_gain_dict


def train_CNF(label_file, test_file, eps, K, method):
    print eps
    print K

    start = time.clock()

    def CNF_gain_1(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        if n == 0:
            return 0
        else:
            return n * (math.log(n/(p+n)) - math.log(N/(P+N)))

    def CNF_gain_3(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        if n/N < 0.5:
            return 0
        else:
            return (n+p) / (N+P)

    def CNF_gain_2(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        return n / (p + 0.0001)

    def learn_neg_conj_terms(pairs, pos_idx, neg_idx, p, k):
        terms = list()
        terms.append([p])

        cur_term = [p]
        cur_L = pairs[p] == True
        cols = copy.deepcopy(generate_label_for_blocking.COLS_TO_USE_APPENDED)
        cols.remove(p)

        print cur_term

        for i in range(k):
            selected_p = ""
            max_gain = 0
            #print cur_term
            for cand_p in cols:
                conj_L = cur_L & (pairs[cand_p] == True)
                if method == 1:
                    cur_gain = CNF_gain_1(labeled_pairs, conj_L, pos_idx, neg_idx)
                elif method == 2:
                    cur_gain = CNF_gain_2(labeled_pairs, conj_L, pos_idx, neg_idx)
                else:
                    cur_gain = CNF_gain_3(labeled_pairs, conj_L, pos_idx, neg_idx)
                if cur_gain > max_gain:
                    max_gain = cur_gain
                    selected_p = cand_p

            if len(selected_p) > 0:
                #print selected_p
                cols.remove(selected_p)
                cur_term.append(selected_p)
                terms.append(copy.deepcopy(cur_term))
                cur_L &= (pairs[selected_p] == True)
                print cur_term
            else:
                break
        return terms

    print 'training CNF blocking....'

    if method == 1:
        cand_file_path = label_file[:-4] + '_predicates_CNF_1.pkl'
    elif method == 2:
        cand_file_path = label_file[:-4] + '_predicates_CNF_2.pkl'
    else:
        cand_file_path = label_file[:-4] + '_predicates_CNF_3.pkl'

    labeled_pairs = pd.read_pickle(label_file)
    cols_to_use = generate_label_for_blocking.COLS_TO_USE_APPENDED
    #cols_to_use = generate_label_for_blocking.COLS_TO_USE

    Pos = labeled_pairs['match'] == True
    Neg = ~Pos

    for col in cols_to_use:
        labeled_pairs[col] = ~labeled_pairs[col]

    target_num = (1-eps) * len(labeled_pairs[Pos] == True)
    CNF = []
    cand_terms = []

    if not os.path.exists(cand_file_path):
        for col in cols_to_use:
            cand_terms.extend(learn_neg_conj_terms(labeled_pairs, Pos, Neg, col, K-1))
        print cand_terms
        with open(cand_file_path, 'wb') as fp:
            pickle.dump(cand_terms, fp)
    else:
        with open(cand_file_path, 'rb') as fp:
            cand_terms_load = pickle.load(fp)
        for term in cand_terms_load:
            if len(term) <= K:
                cand_terms.append(term)

    limit_pos_samples_removed = len(labeled_pairs[Pos] == True) * eps * 0.5

    conjunction_term = ['ln_f', 'fn_1']
    #conjunction_term = ['ln_f', 'mn_1']

    for term in conjunction_term:
        L = pd.Series([True] * len(labeled_pairs))
        L &= labeled_pairs[term] == True
        Pos &= ~L
        Neg &= ~L

    while len(labeled_pairs[Pos] == True) > target_num:
        max_val = 0
        max_term = []
        cur_target_num = (len(labeled_pairs[Pos] == True) - target_num) * 2

        for term in cand_terms:
            L = pd.Series([True] * len(labeled_pairs))
            for predicate in term:
                L &= (labeled_pairs[predicate] == True)
            #if len(labeled_pairs[L & Pos] == True) > limit_pos_samples_removed:
            if len(labeled_pairs[L & Pos] == True) > cur_target_num:
                continue
            if method == 1:
                cur_val = CNF_gain_1(labeled_pairs, L, Pos, Neg)
            elif method == 2:
                cur_val = CNF_gain_2(labeled_pairs, L, Pos, Neg)
            else:
                cur_val = CNF_gain_3(labeled_pairs, L, Pos, Neg)
            #print cur_val
            if cur_val > max_val:
                max_val = cur_val
                max_term = term

        if max_val > 0.0:
            # find rows that is true for max term
            max_L = pd.Series([True] * len(labeled_pairs))
            for predicate in max_term:
                max_L &= labeled_pairs[predicate] == True

            CNF.append(max_term)
            print CNF
            Pos = Pos & ~max_L
            Neg = Neg & ~max_L

            cand_terms.remove(max_term)
        else:
            break

    print('*****\n')
    print CNF
    #
    # pos_pairs = labeled_pairs['match'] == True
    #
    # RR = 1.0 - (len(labeled_pairs[Pos] == True) + len(labeled_pairs[Neg] == True)) / \
    #      float(len(labeled_pairs))
    # PC = len(labeled_pairs[Pos] == True) / float(len(labeled_pairs[pos_pairs]))
    # F = 2 * RR * PC / (RR + PC)
    #
    # print "RR : " + str(RR)
    # print "PC : " + str(PC)
    # print "F  : " + str(F)

    #labeled_pairs = pd.read_pickle(label_file)
    #print ('****** train ******\n')
    #calc_stat(labeled_pairs, CNF, 'CNF')

    end = time.clock()
    print "learning time : " + str(end-start)

    start = time.clock()
    print ('****** test ******\n')
    test_pairs = pd.read_pickle(test_file)
    calc_stat(test_pairs, CNF, 'CNF')
    end = time.clock()
    print "blocking time : " + str(end-start)


def train_DNF(label_file, test_file, eps, K):
    start = time.clock()
    print eps
    print K
    def DNF_gain(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        #print P
        #print N
        #print p
        #print n
        if p == 0:
            return 0
        else:
            return p * (math.log(p/(p+n)) - math.log(P/(P+N)))

    def DNF_gain_2(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        if n == 0:
            return 0
        else:
            return p / n

    def DNF_gain_3(pairs, term_idx, pos_idx, neg_idx):
        P = float(len(pairs[pos_idx]))
        N = float(len(pairs[neg_idx]))
        p = float(len(pairs[pos_idx & term_idx]))
        n = float(len(pairs[neg_idx & term_idx]))
        if p/P < 0.5:
            return 0
        else:
            return 1 - (n+p) / (N+P)

    def learn_pos_conj_terms(pairs, pos_idx, neg_idx, p, k):
        terms = []
        terms.append([p])

        cur_term = [p]
        cur_L = pairs[p] == True
        cols = copy.deepcopy(generate_label_for_blocking.COLS_TO_USE_APPENDED)
        cols.remove(p)

        print cur_term

        for i in range(k):
            selected_p = ""
            max_gain = 0
            for cand_p in cols:
                conj_L = cur_L & (pairs[cand_p] == True)
                cur_gain = DNF_gain(labeled_pairs, conj_L, pos_idx, neg_idx)
                if cur_gain > max_gain:
                    max_gain = cur_gain
                    selected_p = cand_p

            if len(selected_p) > 0:
                cols.remove(selected_p)
                cur_term.append(selected_p)
                terms.append(copy.deepcopy(cur_term))
                cur_L &= (pairs[selected_p] == True)
                print cur_term
            else:
                break
        return terms

    print 'training DNF blocking ...'
    labeled_pairs = pd.read_pickle(label_file)
    cols_to_use = generate_label_for_blocking.COLS_TO_USE_APPENDED

    cand_file_path = label_file[:-4] + '_predicates_DNF.pkl'

    Pos = labeled_pairs['match'] == True
    Neg = ~Pos

    target_num = eps * len(labeled_pairs[Pos] == True)
    DNF = []
    cand_terms = []

    if not os.path.exists(cand_file_path):
        for col in cols_to_use:
            cand_terms.extend(learn_pos_conj_terms(labeled_pairs, Pos, Neg, col, K - 1))
        print cand_terms
        with open(cand_file_path, 'wb') as fp:
            pickle.dump(cand_terms, fp)
    else:
        with open(cand_file_path, 'rb') as fp:
            cand_terms_load = pickle.load(fp)
        for term in cand_terms_load:
            if len(term) <= K:
                # cur_cov =  pd.Series([True] * len(labeled_pairs))
                # for predicate in term:
                #     cur_cov &= (labeled_pairs[predicate] == True)
                #     if len(labeled_pairs[cur_cov] == True) < 0.5 * len(labeled_pairs):
                #         cand_terms.append(term)
                cand_terms.append(term)

    print 'loading predicates done'

    # limited_num_cols = int(len(generate_label_for_blocking.COLS_TO_USE_APPENDED) * 0.7)
    # filtered_count = 0
    # cnt = 0
    # for i in range(len(labeled_pairs)):
    #     cnt += 1
    #     if cnt % 1000 == 0:
    #         print cnt
    #     if labeled_pairs.loc[i]['match'] == False and \
    #                     labeled_pairs.loc[i][generate_label_for_blocking.COLS_TO_USE_APPENDED].isin([False]).sum() \
    #                     > limited_num_cols:
    #         Neg.loc[i] = False
    #         filtered_count += 1
    # print "filtered count : " + str(filtered_count)

    limit_neg_samples_removed = len(labeled_pairs[Neg] == True) * 0.1
    limit_pos_samples_remain = len(labeled_pairs[Pos] == True) * 0.5

    while len(labeled_pairs[Pos] == True) > target_num:
        print len(labeled_pairs[Pos])
        max_val = 0
        max_term = []

        for term in cand_terms:
            L = pd.Series([True] * len(labeled_pairs))
            for predicate in term:
                L &= (labeled_pairs[predicate] == True)
            if len(labeled_pairs[L & Pos] == True) < 0.1 * len(labeled_pairs[Pos] == True):
            #if len(labeled_pairs[L & Neg]) > len(labeled_pairs[Neg]) * 0.3:
            #len(labeled_pairs[L & Pos] == True) < 0.1 * len(labeled_pairs[Pos] == True):
                continue

            cur_val = DNF_gain_2(labeled_pairs, L, Pos, Neg)
            #print cur_val
            if cur_val > max_val:
                max_val = cur_val
                max_term = term

        if max_val > 0.0:
            # find rows that is true for max term
            max_L = pd.Series([True] * len(labeled_pairs))
            for predicate in max_term:
                max_L &= labeled_pairs[predicate] == True

            DNF.append(max_term)
            print DNF
            Pos &= (~max_L)
            Neg &= (~max_L)

            cand_terms.remove(max_term)
        else:
            break

    print('*****\n')
    print DNF

    pos_pairs = labeled_pairs['match'] == True

    # RR = (len(labeled_pairs[Pos] == True) + len(labeled_pairs[Neg] == True)) / \
    #      float(len(labeled_pairs))
    # PC = 1.0 - len(labeled_pairs[Pos] == True) / float(len(labeled_pairs[pos_pairs]))
    # F = 2 * RR * PC / (RR + PC)
    #
    # print "RR : " + str(RR)
    # print "PC : " + str(PC)
    # print "F  : " + str(F)

    #labeled_pairs = pd.read_pickle(label_file)
    #print ('****** train ******\n')
    #calc_stat(labeled_pairs, DNF, 'DNF')
    end = time.clock()
    print "learning time : " + str(end-start)

    start = time.clock()
    print ('****** test ******\n')
    test_pairs = pd.read_pickle(test_file)
    calc_stat(test_pairs, DNF, 'DNF')
    end = time.clock()
    print "blocking time : " + str(end - start)

if __name__ == '__main__':
    if sys.argv[3] == 'CNF':
        train_CNF(sys.argv[1], sys.argv[2], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        train_CNF(sys.argv[2], sys.argv[1], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[3] == 'DNF':
        train_DNF(sys.argv[1], sys.argv[2], float(sys.argv[4]), int(sys.argv[5]))
        train_DNF(sys.argv[2], sys.argv[1], float(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[3] == 'comp':
        rank_keys(sys.argv[1])
    else:
        print 'Usage : train_blocking.py [label file] [test file] [CNF/DNF] [eps] [K]'