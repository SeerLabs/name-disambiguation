import glob
import random
import timeit
import MySQLdb
import numpy as np
import json
from tqdm import tqdm
from sklearn.externals import joblib
import sys

from util import config, utils
from classifier.feature_vector_bow_compact import SimilarityProfileBoW as SimilarityProfile
from dao.pubmed_cluster import AuthorCluster
from dao.author_block import AuthorPoint
from dao.query_doc import QueryDoc

model = joblib.load(config.MODEL_FILE)

RECORD_EPS = 0.4454
CLUSTER_EPS = 0.7260
#CLUSTER_EPS = 0.90
K = 10

CANNAMES_QUERY = "SELECT id FROM cannames WHERE fname like '%s' AND lname = '%s'"
AUTHORS_QUERY = "SELECT id FROM authors WHERE cluster = %s"


def record_based_query_ours(input_filename, order=1):
    with open(input_filename) as fp:
        input = json.load(fp)
    if input is not None:
        ret_list = list()
        cid_aids_dict = dict()
        cid_cluster_dict = dict()

        # make the query author point
        query_doc = QueryDoc(input, order)
        query_author = query_doc.get_author()
        # put temp id 0
        query_point = AuthorPoint(0, query_author, query_doc)

        first_init = query_author.get_first_name()[0]
        last_name = query_author.get_last_name()

        # connect db
        db = utils.connect_db()
        cursor = db.cursor()

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
        else:
            return ret_list

        # calculate correspondence
        feature_matrix = np.zeros(shape=(len(cid_cluster_dict), 8), dtype=float)
        mat_idx = 0

        matched_clusters = list()
        cid_list = cid_cluster_dict.keys()

        # calculate cluster-wise distance
        for cid, cluster in cid_cluster_dict.items():
            feat_vector = SimilarityProfile.calc_feature_vector(query_point, cluster)
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
                if not p.get_author().is_compatible(query_point.get_author()):
                    continue
                else:
                    feat_vector = SimilarityProfile.calc_feature_vector(query_point, p)
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
        # top_K_cid = [i[0] for i in matched_points[:K]]
        for point in matched_points:
            ret_list.append(point[0])
        print ret_list
        return ret_list

    else:
        return -1


if __name__ == '__main__':
    record_based_query_ours(sys.argv[1], int(sys.argv[2]))
