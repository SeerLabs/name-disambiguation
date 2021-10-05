import random
import timeit
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import operator

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

CANNAMES_QUERY = "SELECT id FROM cannames WHERE fname like '%s' AND lname = '%s'"
AUTHORS_QUERY = "SELECT id FROM authors WHERE cluster = %s"

total_query_time = 0
total_query_time_ours = 0
total_query_time_rerank_ours = 0

current_query_time = 0
current_query_time_ours = 0
current_query_time_rerank_ours = 0

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

    for aid in query_list:
        cur_label = data_aid_label_dict.get(aid)
        query_aid_label_dict[aid] = cur_label

        data_label_aid_dict.get(cur_label).remove(aid)
        if len(data_label_aid_dict.get(cur_label)) == 0:
            del data_label_aid_dict[cur_label]

    del data_label_aid_dict
    del data_aid_label_dict

    query_sorted_list = sorted(query_aid_label_dict.items(), key=operator.itemgetter(1))
    print query_sorted_list

    print 'run query...'
    # query 1 - original method (record to record)
    cid_aids_dict = dict()
    cid_cluster_dict = dict()

    prev_first_init = ""
    prev_last_name = ""

    for query_aid, query_label in tqdm(query_sorted_list):
        # make author point for the query id
        doc = PubmedDoc(cursor, query_aid, False)
        auth = doc.get_author()
        p_query = AuthorPoint(query_aid, auth, doc)
        matched_points = list()

        first_init = auth.get_first_name()[0]
        last_name = auth.get_last_name()

        if prev_first_init != first_init or prev_last_name != last_name:
            cid_aids_dict = dict()
            cid_cluster_dict = dict()

            cursor.execute(CANNAMES_QUERY % (first_init, last_name))
            cids = cursor.fetchall()

            if len(cids) > 0:
                for cid_instance in cids:
                    cur_cid = str(cid_instance[0])
                    cursor.execute(AUTHORS_QUERY % (cur_cid))
                    aids = cursor.fetchall()
                    if len(aids) > 0:
                        cur_aids = list()
                        for aid_instance in aids:
                            cur_aids.append(int(aid_instance[0]))
                        cid_aids_dict[cur_cid] = cur_aids

                # make clusters
                print "import clusters...."
                for cid, aid_list in tqdm(cid_aids_dict.items()):
                    cur_cluster = AuthorCluster(cursor, aid_list)
                    cid_cluster_dict[cid] = cur_cluster
                    #print cid_cluster_dict[cid].var_names

            prev_first_init = first_init
            prev_last_name = last_name

        start = timeit.default_timer()
        for cid, cluster in cid_cluster_dict.items():
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
        end = timeit.default_timer()
        current_query_time += end - start


    # query 2 - new method (cluster to record)
    for query_aid, query_label in tqdm(query_sorted_list):
        # make author point for the query id
        doc = PubmedDoc(cursor, query_aid, False)
        auth = doc.get_author()
        p_query = AuthorPoint(query_aid, auth, doc)
        matched_points = list()

        first_init = auth.get_first_name()[0]
        last_name = auth.get_last_name()

        if prev_first_init != first_init or prev_last_name != last_name:
            cid_aids_dict = dict()
            cid_cluster_dict = dict()

            cursor.execute(CANNAMES_QUERY % (first_init, last_name))
            cids = cursor.fetchall()

            if len(cids) > 0:
                for cid_instance in cids:
                    cur_cid = str(cid_instance[0])
                    cursor.execute(AUTHORS_QUERY % (cur_cid))
                    aids = cursor.fetchall()
                    if len(aids) > 0:
                        cur_aids = list()
                        for aid_instance in aids:
                            cur_aids.append(int(aid_instance[0]))
                        cid_aids_dict[cur_cid] = cur_aids

                # make clusters
                print "import clusters...."
                for cid, aid_list in tqdm(cid_aids_dict.items()):
                    cur_cluster = AuthorCluster(cursor, aid_list)
                    cid_cluster_dict[cid] = cur_cluster
                    # print cid_cluster_dict[cid].var_names

            prev_first_init = first_init
            prev_last_name = last_name

        start = timeit.default_timer()
        # calculate correspondence
        feature_matrix = np.zeros(shape=(len(cid_cluster_dict), 8), dtype=float)
        mat_idx = 0

        matched_clusters = list()
        cid_list = cid_cluster_dict.keys()

        # calculate cluster-wise distance
        for cid, cluster in cid_cluster_dict.items():
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

        end = timeit.default_timer()
        current_query_time_ours += end - start

        # pointwise re-ranking
        matched_points = list()
        for cluster_tuple in matched_clusters:
            cid = cluster_tuple[0]
            cluster = cid_cluster_dict.get(cid)

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

        # sort the clusters by matched freq.
        matched_points.sort(key=lambda tup: tup[1], reverse=True)

        end = timeit.default_timer()
        current_query_time_rerank_ours += end - start

    print "cur prev time : " + str(current_query_time)
    print "cur our time : " + str(current_query_time_ours)
    print "cur our(reranking) time : " + str(current_query_time_rerank_ours)

    total_query_time += current_query_time
    total_query_time_ours += current_query_time_ours
    total_query_time_rerank_ours += current_query_time_rerank_ours


print '*'*30
print 'final'
print '*'*30
print "cur prev time : " + str(total_query_time)
print "cur our time : " + str(total_query_time_ours)
print "cur our(reranking) time : " + str(total_query_time_rerank_ours)