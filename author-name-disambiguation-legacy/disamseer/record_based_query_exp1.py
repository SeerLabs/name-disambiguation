import random
import timeit
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

from util import config, utils
from classifier.feature_vector_bow_compact import SimilarityProfileBoW as SimilarityProfile
from dao.pubmed_cluster import AuthorCluster
from dao.author_block import AuthorPoint
from dao.pubmed_doc import PubmedDoc

import cPickle as pickle

#PATH = 'medline_top10/input'
PATH = 'dev_list.txt'

# connect db
db = utils.connect_db()
cursor = db.cursor()

model = joblib.load(config.MODEL_FILE)

RECORD_EPS = 0.4454
CLUSTER_EPS = 0.7260
K = 10

TRAIN_RATIO = 0
DEV_RATIO = 1.0
TEST_RATIO = 0

total_query_time = 0
total_num_queries = 0
total_num_queries_with_result = 0
total_correct = 0
avg_rank = 0

total_query_time_ours = 0
total_num_queries_ours = 0
total_num_queries_with_result_ours = 0
total_correct_ours = 0
avg_rank_ours = 0

total_query_time_rerank_ours = 0
total_correct_rerank_ours = 0
avg_rank_rerank_ours = 0
total_num_queries_with_result_rerank_ours = 0

print "open files..."
# file_list = list()
# for filename in glob.glob(PATH + '/*.txt'):
#     file_list.append(filename)

file_list = list()
with open(PATH, 'r') as fp:
    for line in fp:
        file_list.append(line.split('/')[0] + '/input/' + line.split('/')[-1][:-4]+'txt')

random.seed(2)
random.shuffle(file_list)

nfiles = len(file_list)

# train set: 80%, dev set: 10%, test set: 10%
train_list = file_list[:int(nfiles * TRAIN_RATIO)]
dev_list = file_list[int(nfiles * TRAIN_RATIO):int(nfiles * (TRAIN_RATIO + DEV_RATIO))]
test_list = file_list[int(nfiles * (TRAIN_RATIO + DEV_RATIO)):]


result_prev_list = list()
result_prev_not_exist_list = list()

result_new1_list = list()
result_new1_not_exist_list = list()

result_new2_list = list()
result_new2_not_exist_list = list()

file_cnt = 0
# open file list
for fname in dev_list:
    file_cnt += 1
    print str(file_cnt) + ' ' + fname
    print 'loading labels..'
    # key: aid, value: label
    data_aid_label_dict = dict()
    # key: label, value: aid (represent clusters)
    data_label_aid_dict = dict()

    # query aid, labels pair
    query_aid_label_dict = dict()
    with open(fname) as fp:
        for line in fp:
            cur_aid = int(line.split('\t')[0].strip())
            cur_label = int(line.split('\t')[1].strip())
            if data_label_aid_dict.get(cur_label) is None:
                data_label_aid_dict[cur_label] = [cur_aid]
            else:
                data_label_aid_dict.get(cur_label).append(cur_aid)
            data_aid_label_dict[cur_aid] = cur_label

    # randomly select 20% of records and remove
    random.seed(2)
    query_list = random.sample(data_aid_label_dict.keys(),
                               int(len(data_aid_label_dict.keys()) * 0.2))

    # query_file_name = fname.split('/')[0] + '/input_query/' + fname.split('/')[-1]
    # with open(query_file_name, 'w') as fp:
    #     for aid in query_list:
    #         fp.write(str(aid)+ '\n')

    for aid in query_list:
        cur_label = data_aid_label_dict.get(aid)
        query_aid_label_dict[aid] = cur_label

        data_label_aid_dict.get(cur_label).remove(aid)
        if len(data_label_aid_dict.get(cur_label)) == 0:
            del data_label_aid_dict[cur_label]

    print 'import clusters...'
    # import clusters
    data_cid_cluster_dict = dict()
    for cid, aids in tqdm(data_label_aid_dict.items()):
        cur_cluster = AuthorCluster(cursor, aids)
        data_cid_cluster_dict[cid] = cur_cluster

    print 'run query...'
    # query 1 - original method (record to record)
    cur_correct = 0
    cur_correct_rerank = 0
    cur_num_queries = 0
    cur_num_queries_with_result = 0
    cur_avg_rank = 0

    start = timeit.default_timer()
    for query_aid, query_cid in tqdm(query_aid_label_dict.items()):
        # make author point for the query id
        doc = PubmedDoc(cursor, query_aid, False)
        auth = doc.get_author()
        p_query = AuthorPoint(query_aid, auth, doc)
        matched_points = list()

        for cid, cluster in data_cid_cluster_dict.items():
            mat_idx = 0
            feature_matrix = np.zeros(shape=(len(cluster.points), 8),
                                      dtype=float)
            for p in cluster.points:
                # check how many points are "match" according to the classifier
                if not p.get_author().is_compatible(p_query.get_author()):
                    continue
                else:
                    feat_vector = SimilarityProfile.calc_feature_vector(p_query, p)
                    feature_matrix[mat_idx] = feat_vector
                    mat_idx += 1

            if mat_idx > 0:
                feature_matrix = feature_matrix[:mat_idx, :]
                dist = model.predict_proba(feature_matrix)[:, 0]
                num_matches = len(np.where(dist <= RECORD_EPS)[0])
                if num_matches > 1 or (len(cluster.points) < 2 and num_matches > 0):
                    matched_points.append((cid, num_matches))

        # sort the clusters by matched freq.
        matched_points.sort(key=lambda tup: tup[1], reverse=True)

        if data_cid_cluster_dict.get(query_cid) is None:
            # first check if the query's original cluster is empty
            # then we should have no matches, check it
            if len(matched_points) == 0:
                cur_correct += 1
                result_prev_list.append(-1)
            else:
                result_prev_list.append(0)
            result_prev_not_exist_list.append(1)
        else:
            result_prev_not_exist_list.append(0)
            # There should be a match in this case
            top_K_cid = [i[0] for i in matched_points[:K]]
            if query_cid in top_K_cid:
                cur_correct += 1
                cur_avg_rank += top_K_cid.index(query_cid) + 1
                result_prev_list.append(top_K_cid.index)
            else:
                cur_avg_rank += K + 1
                result_prev_list.append(K)
            cur_num_queries_with_result += 1

        cur_num_queries += 1

    end = timeit.default_timer()

    if len(query_aid_label_dict) > 0:
        print 'query time : ' + str(end - start)
        print "cur queries : " + str(cur_correct) + ' / ' + str(cur_num_queries) + \
              ' = ' + format(float(cur_correct) / cur_num_queries, '.3f')
        print "cur avg rank : " + format(float(cur_avg_rank) / cur_num_queries_with_result,
                                         '.3f')

        total_query_time += end - start
        total_num_queries += cur_num_queries
        total_num_queries_with_result += cur_num_queries_with_result
        total_correct += cur_correct
        avg_rank += cur_avg_rank

    # query 2 - new method (cluster to record)
    cur_correct = 0
    cur_num_queries = 0
    cur_num_queries_with_result = 0
    cur_avg_rank = 0

    cur_correct_rerank = 0
    cur_avg_rank_rerank = 0
    cur_num_queries_with_result_rerank = 0

    start = timeit.default_timer()
    for query_aid, query_cid in tqdm(query_aid_label_dict.items()):
        # make author point for the query id
        doc = PubmedDoc(cursor, query_aid, False)
        auth = doc.get_author()
        p_query = AuthorPoint(query_aid, auth, doc)
        #q_fname = auth.get_first_name()
        #q_mname = auth.get_middle_name()

        feature_matrix = np.zeros(shape=(len(data_cid_cluster_dict), 8), dtype=float)
        mat_idx = 0

        matched_clusters = list()
        cid_list = data_cid_cluster_dict.keys()

        # calculate cluster-wise distance
        for cid, cluster in data_cid_cluster_dict.items():
            feat_vector = SimilarityProfile.calc_feature_vector(p_query, cluster)
            feature_matrix[mat_idx] = feat_vector
            mat_idx += 1

        if mat_idx > 0:
            feature_matrix = feature_matrix[:mat_idx, :]
            dist = model.predict_proba(feature_matrix)[:, 0]
            cand_clusters = np.where(dist <= CLUSTER_EPS)[0]
            for cand in cand_clusters:
                matched_clusters.append((cid_list[cand], dist[cand]))

            # sort the clusters by matched freq.
            matched_clusters.sort(key=lambda tup: tup[1])

        if data_cid_cluster_dict.get(query_cid) is None:
            if len(matched_clusters) == 0:
                cur_correct += 1
                result_new1_list.append(-1)
            else:
                result_new1_list.append(0)
            result_new1_not_exist_list.append(1)

        else:
            result_new1_not_exist_list.append(0)
            top_K_cid = [i[0] for i in matched_clusters[:K]]
            if query_cid in top_K_cid:
                cur_correct += 1
                cur_avg_rank += top_K_cid.index(query_cid) + 1
                result_new1_list.append(top_K_cid.index)
            else:
                cur_avg_rank += K + 1
                result_new1_list.append(K)
            cur_num_queries_with_result += 1

            end = timeit.default_timer()

        # pointwise re-ranking
        matched_points = list()
        for cluster_tuple in matched_clusters:
            cid = cluster_tuple[0]
            cluster = data_cid_cluster_dict.get(cid)

            mat_idx = 0
            feature_matrix = np.zeros(shape=(len(cluster.points), 8),
                                      dtype=float)

            for p in cluster.points:
                # check how many points are "match" according to the classifier
                if not p.get_author().is_compatible(p_query.get_author()):
                    continue
                else:
                    feat_vector = SimilarityProfile.calc_feature_vector(p_query, p)
                    feature_matrix[mat_idx] = feat_vector
                    mat_idx += 1

            if mat_idx > 0:
                feature_matrix = feature_matrix[:mat_idx, :]
                dist = model.predict_proba(feature_matrix)[:, 0]
                num_matches = len(np.where(dist <= RECORD_EPS)[0])
                if num_matches > 1 or (len(cluster.points) < 2 and num_matches > 0):
                    matched_points.append((cid, num_matches))

        # sort the clusters by matched freq.
        matched_points.sort(key=lambda tup: tup[1], reverse=True)
        # There should be a match in this case
        top_K_cid = [i[0] for i in matched_points[:K]]

        if data_cid_cluster_dict.get(query_cid) is None:
            # first check if the query's original cluster is empty
            # then we should have no matches, check it
            if len(matched_points) == 0:
                cur_correct_rerank += 1
                result_new2_list.append(-1)
            else:
                result_new2_list.append(0)
            result_new2_not_exist_list.append(1)
        else:
            result_new2_not_exist_list.append(0)
            if query_cid in top_K_cid:
                cur_correct_rerank += 1
                cur_avg_rank_rerank += top_K_cid.index(query_cid) + 1
                result_new2_list.append(top_K_cid.index)
            else:
                cur_avg_rank_rerank += K + 1
                result_new2_list.append(K)

        cur_num_queries += 1

    end_rerank = timeit.default_timer()
    if len(query_aid_label_dict) > 0:
        print 'query time : ' + str(end - start)
        print "cur queries : " + str(cur_correct) + ' / ' + str(cur_num_queries) + \
              ' = ' + format(float(cur_correct) / cur_num_queries, '.3f')
        print "cur avg rank : " + format(float(cur_avg_rank) / cur_num_queries_with_result,
                                         '.3f')
        print 'query time after reranking: ' + str(end_rerank - start)
        print "cur queries after reranking: " + str(cur_correct_rerank) + ' / ' + \
              str(cur_num_queries) + ' = ' + format(float(cur_correct_rerank) /
                                                    cur_num_queries, '.3f')
        print "cur avg rank after reranking : " + format(float(cur_avg_rank_rerank)
                                                         / cur_num_queries_with_result,
                                                         '.3f')

        total_num_queries_ours += cur_num_queries
        total_num_queries_with_result_ours += cur_num_queries_with_result

        total_correct_ours += cur_correct
        total_query_time_ours += end - start
        avg_rank_ours += cur_avg_rank

        total_correct_rerank_ours += cur_correct_rerank
        avg_rank_rerank_ours += cur_avg_rank_rerank
        total_query_time_rerank_ours += end_rerank - start

print '\n' + '*'*10 + 'prev method' + '*'*10
print "total num queries : " + str(total_num_queries)
print "total correct queries : " + str(total_correct)
print "total query time : " + str(total_query_time)
print "avg rank : " + format(float(avg_rank) / total_num_queries_with_result, '.3f') + '\n'

print '*'*10 + 'our method' + '*'*10
print "total num queries : " + str(total_num_queries_ours)
print "total correct queries : " + str(total_correct_ours)
print "total query time : " + str(total_query_time_ours)
print "avg rank : " + format(float(avg_rank_ours) /
                             total_num_queries_with_result, '.3f') + '\n'

print "total correct queries after reranking : " + str(total_correct_rerank_ours)
print "total query time after reranking : " + str(total_query_time_rerank_ours)
print "avg rank after reranking: " + format(float(avg_rank_rerank_ours) /
                                            total_num_queries_with_result, '.3f')

pickle.dump(result_prev_list, 'result_prev_list.pkl')
pickle.dump(result_prev_not_exist_list, 'result_prev_not_exist.pkl')
pickle.dump(result_new1_list, 'result_new1_list.pkl')
pickle.dump(result_new1_not_exist_list, 'result_new1_not_exist.pkl')
pickle.dump(result_new2_list, 'result_new2_list.pkl')
pickle.dump(result_new2_not_exist_list, 'result_new2_not_exist.pkl')
