from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import sys
import os
import timeit
import glob
import xgboost
from tqdm import tqdm

import random

from sklearn.metrics import *

from util import config

#
# from scipy.stats import mode
# from sklearn.ensemble.forest import _partition_estimators, _parallel_helper
# from sklearn.tree._tree import DTYPE
# from sklearn.externals.joblib import Parallel, delayed
# from sklearn.utils import check_array
# from sklearn.utils.validation import check_is_fitted

"""
Script to Train Random Forest pairwise classifier
usage : python train_rf.py [train files dir]
"""

# sampling ratio of each label file
SAMPLE_RATIO = 1.0
# number of tree
NUM_TREE = 100
# number of threads to run
NUM_THREADS = 12

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

SAVE_EVAL = False
LEARN_MODEL = True

NFEAT = 11


def run(train_path, test_path):
    train_list = list()
    file_list = list()

    for filename in glob.glob(train_path + '/*.out'):
        train_list.append(filename)
    for filename in glob.glob(test_path + '/*.out'):
        file_list.append(filename)

    #random.seed(2)
    #random.shuffle(file_list)

    nfiles = len(file_list)

    # train set: 80%, dev set: 10%, test set: 10%
    #train_list = file_list[:int(nfiles*TRAIN_RATIO)]
    #dev_list = file_list[int(nfiles*TRAIN_RATIO):int(nfiles*(TRAIN_RATIO+DEV_RATIO))]
    #test_list = file_list[int(nfiles*(TRAIN_RATIO+DEV_RATIO)):]
    #dev_list = file_list[:int(nfiles*0.5)]
    #test_list = file_list[int(nfiles*0.5):]

    dev_list = list()
    test_list = list()
    with open('dev_list.txt') as fp:
        for fname in fp:
            dev_list.append(test_path + '/' + fname.split('/')[-1].strip())
    with open('test_list.txt') as fp:
        for fname in fp:
            test_list.append(test_path + '/' + fname.split('/')[-1].strip())

    # with open('dev_list.txt', 'w') as fp:
    #     for fname in dev_list:
    #         fp.write(fname+'\n')
    # with open('test_list.txt', 'w') as fp:
    #     for fname in test_list:
    #         fp.write(fname + '\n')

    # load labels
    with open(train_list[0], 'r') as fp:
        for line in fp:
            data_label = line.strip().split(',')
            break

    NFEAT = len(data_label)-1

    def get_feature_matrix(file_list):
        data_list = list()
        nrows = 0
        for file_name in tqdm(file_list):
            cur_dataset = np.genfromtxt(file_name, delimiter=',',
                                        dtype='float', skip_header=1)
            if cur_dataset.ndim == 1:
                cur_dataset = np.reshape(cur_dataset, (1, cur_dataset.shape[0]))

            # print dataset_train.shape
            comp = np.load(file_name[:-4] + '_comp.npy')
            comp = comp.astype(np.float32)
            comp = np.reshape(comp, (comp.shape[0], 1))

            cur_dataset = np.hstack((cur_dataset, comp))

            nrows += cur_dataset.shape[0]
            data_list.append(cur_dataset)

        dataset = np.zeros((nrows, NFEAT+2), dtype=np.float32)
        cur_idx = 0
        for data in data_list:
            dataset[cur_idx:cur_idx+data.shape[0]] = data
            cur_idx += data.shape[0]
        return dataset

    if LEARN_MODEL:
        print 'loading train'
        dataset_train = get_feature_matrix(train_list)
        #dim = len(data_label) - 1
        model = xgboost.XGBClassifier(n_estimators=NUM_TREE, nthread=NUM_THREADS)
        #model = RandomForestClassifier(n_estimators=NUM_TREE, n_jobs=NUM_THREADS)

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        X_training = dataset_train[:, :-2]
        comp = dataset_train[:, -1]
        y_training = dataset_train[:, -2]

        #print X_training.shape, comp.shape, y_training.shape
        #print X_test.shape, comp_test.shape, y_test.shape

        ros = RandomOverSampler()
        #ros = RandomUnderSampler()
        X_training, y_training = ros.fit_sample(X_training, y_training)

        print len(np.where(y_training == 0)[0])
        print len(np.where(y_training == 1)[0])

        print "fitting data.."
        start = timeit.default_timer()
        model.fit(X_training, y_training)
        end = timeit.default_timer()

        print "train time: ", end-start

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # oob_score = rf.oob_score_
        # print("oob score: " + str(oob_score))

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(NFEAT):
            print("%d. %s (%f)" % (f + 1, data_label[indices[f]], importances[indices[f]]))

        joblib.dump(model, config.MODEL_FILE)

    else:
        model = joblib.load(config.MODEL_FILE)

    print 'loading dev'
    dataset_dev = get_feature_matrix(dev_list)
    print 'loading test'
    dataset_test = get_feature_matrix(test_list)

    X_test = dataset_dev[:, :-2]
    # X_test = np.delete(X_test, 2, 1)
    # print X_test.shape[0]

    comp_test = dataset_dev[:, -1]
    y_test = dataset_dev[:, -2]

    start = timeit.default_timer()

    #print idx_to_check_train

    print "predict data.."

    if LEARN_MODEL:
        idx_to_check_train = np.where(comp == 1.0)
        result_train_filtered = np.zeros(y_training.shape)
        result_train_filtered_prob = np.zeros((y_training.shape[0], 2))

        result_train_prob = model.predict_proba(X_training)
        result_train = model.predict(X_training)

        result_train_filtered_prob[idx_to_check_train] = \
            model.predict_proba(X_training[idx_to_check_train])
        result_train_filtered[idx_to_check_train] = \
            model.predict(X_training[idx_to_check_train])

    idx_to_check_test = np.where(comp_test == 1.0)
    #print idx_to_check_test

    result_test_filtered = np.zeros(y_test.shape)
    result_test_filtered_prob = np.zeros((y_test.shape[0], 2))

    result_test_prob = model.predict_proba(X_test)
    result_test = model.predict(X_test)

    result_test_filtered_prob[idx_to_check_test] = \
        model.predict_proba(X_test[idx_to_check_test])
    result_test_filtered[idx_to_check_test] = \
        model.predict(X_test[idx_to_check_test])

    end = timeit.default_timer()

    print "eval time: ", end - start

    if SAVE_EVAL:
        if LEARN_MODEL:
            np.save('result_model/xgb_prev_train.npy',
                    precision_recall_curve(y_training, result_train_prob[:, 1]))
            np.save('result_model/xgb_prev_filtered_train.npy',
                    precision_recall_curve(y_training, result_train_filtered_prob[:, 1]))

        np.save('result_model/rf_query_test.npy',
                precision_recall_curve(y_test, result_test_prob[:, 1]))
        np.save('result_model/rf_query_filtered_test.npy',
                precision_recall_curve(y_test, result_test_filtered_prob[:, 1]))

    print("classification report")
    if LEARN_MODEL:
        print(classification_report(y_training, result_train, digits=4))
        print(classification_report(y_training, result_train_filtered, digits=4))

    print(classification_report(y_test, result_test, digits=4))
    print(classification_report(y_test, result_test_filtered, digits=4))


    # X = dataset[:, :-2]
    # comp = dataset[:, -2:-1]
    # y = dataset[:, -1]
    #
    # skf = StratifiedKFold(n_splits=5, random_state=2)
    # # stat_train = [[0 for x in range(2)] for x in range(2)]
    # # stat_test = [[0 for x in range(2)] for x in range(2)]
    #
    # train_time = 0
    # eval_time = 0
    #
    # # sum_train_result = None
    # sum_train_result_prob = None
    # # sum_test_result = None
    # sum_test_result_prob = None
    #
    # # sum_train_result_prob_before = None
    # # sum_test_result_prob_before = None
    #
    # sum_train_label = None
    # sum_test_label = None
    #
    # baseline_train_result = None
    # baseline_test_result = None
    #
    # idx = 0
    # for train, test in skf.split(X, y):
    #     print 'fold', idx
    #     idx += 1
    #
    #     cur_train_X = X[train]
    #     cur_train_y = y[train]
    #
    #     cur_test_X = X[test]
    #     cur_test_y = y[test]
    #
    #     # result_train = np.zeros(cur_train_y.shape)
    #     result_train_prob = np.zeros((cur_train_y.shape[0], 2))
    #     # result_test = np.zeros(cur_test_y.shape)
    #     result_test_prob = np.zeros((cur_test_y.shape[0], 2))
    #     result_train_prob[:, 0] = 1.0
    #     result_test_prob[:, 0] = 1.0
    #
    #     # result_train = np.zeros(cur_train_y.shape)
    #     # result_train_prob_before = np.zeros((cur_train_y.shape[0], 2))
    #     # result_test = np.zeros(cur_test_y.shape)
    #     # result_test_prob_before = np.zeros((cur_test_y.shape[0], 2))
    #
    #     cur_baseline_train = np.zeros(cur_train_y.shape)
    #     cur_baseline_test = np.zeros(cur_test_y.shape)
    #
    #     start = timeit.default_timer()
    #     model.fit(cur_train_X, cur_train_y)
    #     end = timeit.default_timer()
    #     train_time += (end - start)
    #
    #     start = timeit.default_timer()
    #
    #     cur_train_comp = comp[train]
    #     cur_test_comp = comp[test]
    #
    #     idx_to_check_train = np.where(cur_train_comp == 1.0)[0]
    #     idx_to_check_test = np.where(cur_test_comp == 1.0)[0]
    #
    #     cur_baseline_train[idx_to_check_train] = 1.0
    #     cur_baseline_test[idx_to_check_test] = 1.0
    #
    #     result_train_prob_filtered = \
    #         model.predict_proba(cur_train_X[idx_to_check_train])
    #     # result_train_filtered = model.predict(cur_train_X[idx_to_check_train])
    #
    #     result_train_prob[idx_to_check_train] = result_train_prob_filtered
    #     # result_train[idx_to_check_train] = result_train_filtered
    #
    #     result_test_prob_filtered = \
    #         model.predict_proba(cur_test_X[idx_to_check_test])
    #     # result_test_filtered = model.predict(cur_test_X[idx_to_check_test])
    #
    #     result_test_prob[idx_to_check_test] = result_test_prob_filtered
    #     # result_test[idx_to_check_test] =
    #
    #     end = timeit.default_timer()
    #     eval_time += (end - start)
    #
    #     if sum_train_result_prob is None:
    #         # sum_train_result = result_train
    #         sum_train_result_prob = result_train_prob
    #         # sum_test_result = result_test
    #         sum_test_result_prob = result_test_prob
    #         sum_train_label = cur_train_y
    #         sum_test_label = cur_test_y
    #
    #         # sum_train_result_prob_before = result_train_prob_before
    #         # sum_test_result_prob_before = result_test_prob_before
    #
    #         baseline_train_result = cur_baseline_train
    #         baseline_test_result = cur_baseline_test
    #     else:
    #         # sum_train_result = np.concatenate((sum_train_result, result_train))
    #         sum_train_result_prob = np.concatenate((sum_train_result_prob, result_train_prob))
    #         # sum_test_result = np.concatenate((sum_test_result, result_test))
    #         sum_test_result_prob = np.concatenate((sum_test_result_prob, result_test_prob))
    #         sum_train_label = np.concatenate((sum_train_label, cur_train_y))
    #         sum_test_label = np.concatenate((sum_test_label, cur_test_y))
    #
    #         # sum_train_result_prob_before = np.concatenate((sum_train_result_prob_before, result_train_prob_before))
    #         # sum_test_result_prob_before = np.concatenate((sum_test_result_prob_before, result_test_prob_before))
    #
    #         baseline_train_result = np.concatenate((baseline_train_result, cur_baseline_train))
    #         baseline_test_result = np.concatenate((baseline_test_result, cur_baseline_test))
    #
    #     # for i in range(len(result_train)):
    #     #     decision = 1 if result_train[i] >= 0.9 else 0
    #     #     stat_train[decision][int(cur_train_y[i])] += 1
    #     #
    #     # for i in range(len(result_test)):
    #     #     decision = 1 if result_test[i] >= 0.9 else 0
    #     #     stat_test[decision][int(cur_test_y[i])] += 1
    #
    # print '--------'
    # print 'total train time : ' + str(train_time) + ' s'
    # print 'total predict time : ' + str(eval_time) + ' s'
    #
    # np.save('rf_300_original_removed_train.npy',
    #         precision_recall_curve(sum_train_label, sum_train_result_prob[:, 1]))
    # np.save('rf_300_original_removed_test.npy',
    #         precision_recall_curve(sum_test_label, sum_test_result_prob[:, 1]))
    # # np.save('xgboost_300_w2v_tfidf_train.npy',
    # #         precision_recall_curve(sum_train_label, sum_train_result_prob_before[:, 1]))
    # # np.save('xgboost_300_w2v_tfidf_test.npy',
    # #         precision_recall_curve(sum_test_label, sum_test_result_prob_before[:, 1]))
    # # np.save('baseline.npy',
    # #         precision_recall_curve(sum_train_label, baseline_train_result))

    # print '---- train ----'
    # print '00:', stat_train[0][0]
    # print '01:', stat_train[0][1]
    # print '10:', stat_train[1][0]
    # print '11:', stat_train[1][1]
    #
    # precision = float(stat_train[1][1]) / (stat_train[1][1] + stat_train[1][0])
    # recall = float(stat_train[1][1]) / (stat_train[1][1] + stat_train[0][1])
    # f1 = 2 * (precision * recall) / (precision + recall)
    #
    # print 'precision:', precision
    # print 'recall:', recall
    # print 'fscore:', f1
    #
    # print '---- test ----'
    # print '00:', stat_test[0][0]
    # print '01:', stat_test[0][1]
    # print '10:', stat_test[1][0]
    # print '11:', stat_test[1][1]
    #
    # precision = float(stat_test[1][1]) / (stat_test[1][1] + stat_test[1][0])
    # recall = float(stat_test[1][1]) / (stat_test[1][1] + stat_test[0][1])
    # f1 = 2 * (precision * recall) / (precision + recall)
    #
    # print 'precision:', precision
    # print 'recall:', recall
    # print 'fscore:', f1
    #
    # print '--------'
    # print 'total predict time : ' + str(total_time) + ' s'

    # if test_path is not None:
    #     testset = None
    #     for dirpath, dirs, files in os.walk(test_path):
    #         for testfile in files:
    #             test_list.append(os.path.join(test_path, testfile))
    #
    #     for testfile in test_list:
    #         curdata = genfromtxt(open(testfile, 'r'), delimiter=',', dtype='float')[1:]
    #         if testset is not None:
    #             testset = numpy.concatenate((testset, curdata), axis=0)
    #         else:
    #             testset = numpy.copy(curdata)
    #
    #     label = [x[dim - 1] for x in testset]
    #     test = [x[0:dim - 1] for x in testset]
    #
    #     result = rf.predict(test)
    #     stat = [[0 for x in range(2)] for x in range(2)]
    #
    #     for i in range(0, len(result)-1):
    #         stat[int(result[i])][int(label[i])] += 1
    #
    #     print '00:', stat[0][0]
    #     print '01:', stat[0][1]
    #     print '10:', stat[1][0]
    #     print '11:', stat[1][1]
    #
    #     precision = float(stat[1][1])/(stat[1][1] + stat[1][0])
    #     recall = float(stat[1][1])/(stat[1][1] + stat[0][1])
    #     f1 = 2 * (precision * recall) / (precision + recall)
    #
    #     print 'precision:', precision
    #     print 'recall:', recall
    #     print 'fscore:', f1

    # joblib.dump(rf, DUMP_FILE_NAME)


if __name__ == "__main__":
    trainPath = sys.argv[1]
    testPath = sys.argv[2]
    run(trainPath, testPath)
