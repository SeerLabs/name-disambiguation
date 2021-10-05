import cPickle as pickle
import gc

import MySQLdb
import numpy
from scipy import sparse
from sklearn.externals import joblib

from util import config
from util import utils
#from classifier.feature_vector_bow_compact import SimilarityProfileBoW as SimilarityProfile
from classifier.feature_vector_bow import SimilarityProfileBoW as SimilarityProfile
#from classifier.feature_vector import SimilarityProfile
from author_point import AuthorPoint
from parse_error import NameParseError
from pubmed_doc import PubmedDoc


class AuthorBlock(object):
    """
    Class that Loads the records within the block
    """

    def __init__(self, inputAids, isFile):
        self.rf = None
        # list that contains all author ids
        self.aids = []
        # dict that contains all author points (author points = author + doc)
        self.points = {}
        self.nauthors = 0
        self.dist_matrix = None
        #self.regions = {}
        # self.distCache = {}

        # connect db
        db = utils.connect_db()
        cursor = db.cursor()

        if isFile:
            # load from file
            with open(inputAids) as fp:
                for line in fp:
                    self.aids.append(int(line))
        else:
            # load from array
            self.aids.extend(inputAids)

        i = 0

        # count the number with parssing errors
        exception_count = 0
        aids_to_remove = []

        for aid in self.aids:
            i += 1
            #if i % 100 == 0:
            #    print "Loaded " + str(i)
            try:
                doc = PubmedDoc(cursor, aid, False)
            except NameParseError as e:
                print e
                exception_count += 1
                aids_to_remove.append(aid)
                continue

            auth = doc.get_author_by_id(aid)
            point = AuthorPoint(aid, auth, doc)
            self.points[aid] = point

        for aid in aids_to_remove:
            self.aids.remove(aid)

        self.nauthors = len(self.aids)
        # print 'Loading block finished..'

    def get_all_aids(self):
        return self.aids

    def get_points(self):
        return self.points

    def get_distance_matrix(self):
        return self.dist_matrix

    def get_nauthors(self):
        return self.nauthors

    # experiment code for DNN. Needs to be refactored
    def calc_BoW_features(self, filename, gold_dic):
        labels = list()
        titles = list()
        affils = list()
        journals = list()
        abstracts = list()

        ndata = len(self.aids)

        # make numpy matricies for metadata with multiple values
        # grant_array = numpy.zeros((ndata, len(utils.GRANT_DICT.keys())))
        label_array = numpy.zeros((ndata, 1), dtype=numpy.int32)
        chemical_array = sparse.dok_matrix((ndata, len(disamseer.util.utils.CHEMICAL_DICT.keys())), dtype=numpy.int32)
        mesh_array = sparse.dok_matrix((ndata, len(disamseer.util.utils.MESH_DICT.keys())), dtype=numpy.int32)
        name_array = sparse.dok_matrix((ndata, len(disamseer.util.utils.NAME_DICT.keys())), dtype=numpy.int32)

        #chemical_array = numpy.zeros((ndata, len(utils.CHEMICAL_DICT.keys())), dtype=numpy.int32)
        #mesh_array = numpy.zeros((ndata, len(utils.MESH_DICT.keys())), dtype=numpy.int32)
        #name_array = numpy.zeros((ndata, len(utils.NAME_DICT.keys())), dtype=numpy.int32)

        # get list of labels and metadata
        for i in range(len(self.aids)):
            aid = self.aids[i]
            label_array[i] = int(gold_dic.get(aid))

            cur_author_point = self.points.get(aid)
            cur_doc = cur_author_point.get_doc()
            cur_author = cur_author_point.get_author()

            titles.append(cur_doc.get_title())
            affils.append(cur_author.get_affil())
            journals.append(cur_doc.get_venue())
            abstracts.append(cur_doc.get_abstract())

            # for grant in cur_doc.get_grants():
            #     grant_idx = utils.GRANT_DICT.get(grant)
            #     if grant_idx is not None:
            #         grant_array[i, grant_idx] += 1

            for chem in cur_doc.get_chemicals():
                chem_idx = disamseer.util.utils.CHEMICAL_DICT.get(chem)
                if chem_idx is not None:
                    chemical_array[i, chem_idx] += 1

            for mesh in cur_doc.get_meshes():
                mesh_idx = disamseer.util.utils.MESH_DICT.get(mesh)
                if mesh_idx is not None:
                    mesh_array[i, mesh_idx] += 1

            for name in cur_doc.author_names:
                name_idx = disamseer.util.utils.NAME_DICT.get(name)
                if name_idx is not None:
                    name_array[i, name_idx] += 1


        #label_array = numpy.array(labels, dtype=numpy.int32)
        #label_array = label_array[:, numpy.newaxis]

        label_spm = sparse.csr_matrix(label_array, dtype=numpy.int32)
        #grant_spm = sparse.csr_matrix(grant_array)
        chemical_spm = sparse.csr_matrix(chemical_array, dtype=numpy.int32)
        mesh_spm = sparse.csr_matrix(mesh_array, dtype=numpy.int32)
        coauthor_spm = sparse.csr_matrix(name_array, dtype=numpy.int32)

        # write sparse matrix of title
        title_spm = disamseer.util.utils.title_vectorizer.transform(titles)
        affil_spm = disamseer.util.utils.affil_vectorizer.transform(affils)
        journal_spm = disamseer.util.utils.journal_vectorizer.transform(journals)
        abstract_spm = disamseer.util.utils.abstract_vectorizer.transform(abstracts)

        #print len(utils.title_vectorizer.get_feature_names())
        #print utils.title_vectorizer.get_feature_names()
        #print title_spm
        # print sparse.find(title_spm.getrow(0))
        # print sparse.find(affil_spm.getrow(0))
        # print sparse.find(journal_spm.getrow(0))
        # print sparse.find(abstract_spm.getrow(0))
        # print sparse.find(chemical_spm.getrow(0))
        # print sparse.find(mesh_spm.getrow(0))
        # print sparse.find(coauthor_spm.getrow(0))
        #print sparse.find(grant_spm.getrow(0))

        print(title_spm.shape[1])
        print(affil_spm.shape[1])
        print(journal_spm.shape[1])
        print(abstract_spm.shape[1])
        print(chemical_spm.shape[1])
        print(mesh_spm.shape[1])
        print(coauthor_spm.shape[1])

        total_spm = sparse.hstack((title_spm, affil_spm, journal_spm, abstract_spm,
                                   chemical_spm, mesh_spm, coauthor_spm, label_spm))
        # total_spm = sparse.hstack((title_spm, abstract_spm,
        #                            chemical_spm, mesh_spm, coauthor_spm, label_spm))

        del titles
        del affils
        del journals
        del chemical_array
        del mesh_array
        del name_array
        gc.collect()

        sparse.save_npz(filename, total_spm)

    def extract_metadata(self, filename, gold_dic):
        ndata = len(self.aids)
        metadatas = []

        for i in range(ndata):
            cur_doc = self.points.get(self.aids[i]).get_doc()
            cur_title = cur_doc.get_title()
            cur_abstract = cur_doc.get_abstract()
            cur_affil = cur_doc.get_author().get_affil()
            cur_grant = cur_doc.get_grants()
            cur_journal = cur_doc.get_venue()
            cur_chemical = cur_doc.get_chemicals()
            cur_mesh = cur_doc.get_meshes()
            cur_coauthor = cur_doc.author_names
            cur_year = cur_doc.get_year()
            cur_label = gold_dic.get(self.aids[i])

            metadatas.append([cur_title, cur_abstract, cur_affil, cur_grant,
                              cur_journal, cur_chemical, cur_mesh, cur_coauthor,
                              cur_year, cur_label])

        pickle.dump(metadatas, open(filename, 'wb'))

    def calc_word2vec_features(self, feature_filename, label_filename, gold_dic):
        ndata = len(self.aids)
        max_title_size = 200
        w2v_dim = 300

        label_array = numpy.zeros((ndata, 1), dtype=numpy.int32)
        w2v_array = numpy.zeros((ndata*max_title_size, 300), dtype=numpy.float16)
        for i in range(ndata):
            cur_author_point = self.points.get(self.aids[i])
            cur_doc = cur_author_point.get_doc()
            title_cleaned = disamseer.util.utils.remove_punc(cur_doc.get_title())
            title_list = [token for token in title_cleaned.split() if token not in disamseer.util.utils.STOP_WORDS]
            for j in range(len(title_list)):
                token = title_list[j]
                if token in disamseer.util.utils.W2V_MODEL.wv.vocab:
                    w2v_array[i * max_title_size + j] = disamseer.util.utils.W2V_MODEL[token]
                if j == max_title_size -1:
                    break
            label_array[i] = int(gold_dic.get(self.aids[i]))

        feature_data = numpy.empty((1, w2v_dim*2), dtype=numpy.float16)
        Y = numpy.empty((1, 1), dtype=numpy.float16)

        #idx = 0
        for i in range(ndata):
            p1 = self.points.get(self.aids[i])

            cur_feature_data = numpy.zeros(((ndata - i) * max_title_size, w2v_dim * 2), dtype=numpy.float16)
            cur_Y = numpy.zeros((ndata - i, 1), dtype=numpy.float16)
            idx = 0

            for j in range(i+1, ndata):
                p2 = self.points.get(self.aids[j])

                if p1.get_author().is_compatible(p2.get_author()):
                    #cur_feature_data = numpy.zeros((max_title_size, w2v_dim*2), dtype=numpy.float16)
                    for k in range(max_title_size):
                        cur_feature_data[idx * max_title_size + k, :w2v_dim] = w2v_array[i * max_title_size + k]
                        cur_feature_data[idx * max_title_size + k, w2v_dim:] = w2v_array[j * max_title_size + k]

                    cur_Y[idx] = 1.0 if label_array[i] == label_array[j] else 0.0
                    idx += 1

            feature_data = numpy.vstack((feature_data, cur_feature_data[:idx*max_title_size]))
            Y = numpy.vstack((Y, cur_Y[:idx]))

        feature_data = feature_data[1:]
        Y = Y[1:]

        joblib.dump(feature_data, feature_filename)
        numpy.save(label_filename, Y)

        #numpy.save(feature_filename, w2v_array)
        #numpy.save(label_filename, label_array)

    def calc_features_and_write(self, fp, gold_dic):
        """
        Calculate features to train the pairwise model
        Please refer feature_extractor.py
        :param fp: file pointer to write
        :param gold_dic: dictionary that contains (aid, label) pairs
        :return: none
        """
        """ Calculate features to train the pairwise model """
        def featureVectorToString(feats):
            featString = ""
            for feat in feats:
                if not featString:
                    featString += str(feat)
                else:
                    featString += ',' + str(feat)
            return featString

        nauthors = len(self.aids)
        num_pairs = int(nauthors * (nauthors-1) * 0.5)
        compatible_array = numpy.zeros(num_pairs, dtype=numpy.int)
        count = 0

        for i in range(nauthors):
            p1 = self.points.get(self.aids[i])
            l1 = gold_dic.get(self.aids[i])
            # f1 = p1.get_author().get_first_name()
            for j in range(i+1, nauthors):
                p2 = self.points.get(self.aids[j])
                l2 = gold_dic.get(self.aids[j])
                # f2 = p2.get_author().get_first_name()

                # if len(f1) == 1 or len(f2) == 1:
                #     f1 = f1[:1]
                #     f2 = f2[:1]

                # if f1 == f2 and AuthorBlock.passConstrains(p1, p2):

                # add compatability
                if p1.get_author().is_compatible(p2.get_author()):
                    compatible_array[count] = 1
                else:
                    compatible_array[count] = 0

                if l1 == l2:
                    feats = SimilarityProfile.calc_feature_vector(p1, p2)
                    fp.write(featureVectorToString(feats) + ',1\n')

                else:
                    feats = SimilarityProfile.calc_feature_vector(p1, p2)
                    fp.write(featureVectorToString(feats) + ',0\n')

                count += 1

        return compatible_array

    def calc_distance_matrix(self):
        """
        calculates pairwise distance matrix (n*n) of each author records
        :return: pairwise distance matrix
        """
        '''
        def comp(a1, a2):
            len1 = len(self.points.get(a1).getAuthor().getFirstName())
            len2 = len(self.points.get(a2).getAuthor().getFirstName())
            return len2 - len1
        '''

        if self.rf is None:
            self.rf = joblib.load(config.MODEL_FILE)

        # since the valuses are [0,1] scale, use float16 to reduce the memory
        self.dist_matrix = numpy.zeros(shape=(self.nauthors, self.nauthors),
                                       dtype=numpy.float16)

        # pct = int(self.nauthors / 10)
        # if pct == 0:
        #     pct = 1

        title1 = ""
        title2 = ""

        # calculate ratio of filtered points
        #num_filtered = 0

        idx_list = []
        loop_counter = 0
        n_vec = self.nauthors * (self.nauthors - 1) / 2
        feature_matrix = numpy.zeros(shape=(n_vec, SimilarityProfile.NFEAT), dtype=float)

        # fill each row
        for i in range(self.nauthors-1):
            #feature_vector_list = []
            #feature_matrix = numpy.zeros(shape=(n_vec, SimilarityProfile.NFEAT), dtype=float)

            # if i % pct == 0:
            #     print str(int(i/pct*10)) + "% done"


            p1 = self.points.get(self.aids[i])
            #namekey = AuthorBlock.getNameKey(p1.getAuthor())
            #namekey_init = namekey[:1]

            for j in range(i+1, self.nauthors):
                p2 = self.points.get(self.aids[j])

                # namekey2 = AuthorBlock.getNameKey(p2.getAuthor())
                # if namekey == namekey2 or namekey_init == namekey2:

                if not p1.get_author().is_compatible(p2.get_author()):
                    self.dist_matrix[i, j] = 0.0
                    #num_filtered += 1
                    continue

                elif p1.get_doc().get_id() == p2.get_doc().get_id():
                    self.dist_matrix[i, j] = 0.0
                    #num_filtered += 1
                    continue

                #elif not p1.get_doc().is_year_compatible(p2.get_doc()):
                #      self.dist_matrix[i, j] = 0.0
                #      #num_filtered += 1
                #      continue

                # calculate feature vectors for each pair
                feat_vector = SimilarityProfile.calc_feature_vector(p1, p2)
                feature_matrix[loop_counter] = feat_vector
                loop_counter += 1
                #feature_vector_list.append(feat)
                idx_list.append((i, j))

        #if len(feature_vector_list) > 0:
        if loop_counter > 0:
            #mat_classifier = numpy.array(feature_vector_list)
            feature_matrix = feature_matrix[:loop_counter, :]
            dist = self.rf.predict_proba(feature_matrix)[:, 1]

            for row in range(loop_counter):
                self.dist_matrix[idx_list[row][0], idx_list[row][1]] = dist[row]

        #print str(num_filtered) + ' / ' + str(len(self.aids) * (len(self.aids)-1) / 2)

        print "start fixing transitivity violations .. "
        # correct transitivity violations
        '''
        for iter in range(5):
             print iter
             changed = False
             for i in range(self.nauthors-2):
                for j in range(i+1, self.nauthors-1):
                    p_ij = self.dist_matrix[i, j]
                    if p_ij == 0.0:
                        continue
                    for k in range(j+1, self.nauthors):
                        p_ik = self.dist_matrix[i, k]
                        p_jk = self.dist_matrix[j, k]
                        if p_ik == 0.0 or p_jk == 0.0:
                            continue
                        if p_ik <= p_ij and p_ik <= p_jk:
                            if p_ij + p_jk - 1.0 > p_ik + 0.05:
                                changed = True
                                w_ij = 1.0 / (p_ij * (1.0 - p_ij))
                                w_ik = 1.0 / (p_ik * (1.0 - p_ik))
                                w_jk = 1.0 / (p_jk * (1.0 - p_jk))

                                den = w_ij*w_jk + w_ij*w_ik + w_jk*w_ik
                                q_ij = (w_ij * (w_jk+w_ik)*p_ij + w_jk*w_ik*(1.0+p_ik-p_jk)) / den
                                q_jk = (w_jk * (w_ij+w_ik)*p_jk + w_ij*w_ik*(1.0+p_ik-p_ij)) / den
                                q_ik = q_ij + q_jk - 1.0

                                p_ij = q_ij
                                self.dist_matrix[i, j] = q_ij
                                self.dist_matrix[i, k] = q_ik
                                self.dist_matrix[j, k] = q_jk
             if not changed:
                 break
        '''
        # make the matrix symmetric
        self.dist_matrix = self.dist_matrix + self.dist_matrix.T
        #return mat_dist

    # get neighbor of idx in the distance matrix
    def get_neighbors(self, idx, eps, min_pts):
        def comp(idx_1, idx_2):
            # idx_check_1 = AuthorBlock.get_idx_in_matrix(idx, idx_1)
            # idx_check_2 = AuthorBlock.get_idx_in_matrix(idx, idx_2)
            idx_check_1 = (idx, idx_1)
            idx_check_2 = (idx, idx_2)
            # idx_check_1 = (idx, idx_1) if idx_1 > idx else (idx_1, idx)
            # idx_check_2 = (idx, idx_2) if idx_2 > idx else (idx_2, idx)
            if self.dist_matrix[idx_check_1] < self.dist_matrix[idx_check_2]:
                return -1
            elif self.dist_matrix[idx_check_1] > self.dist_matrix[idx_check_2]:
                return 1
            else:
                return 0

        neighbors = list()
        # cand_idxes = numpy.concatenate((numpy.where(self.dist_matrix[idx, idx+1:] < eps)[0],
        #                                numpy.where(self.dist_matrix[:idx, idx] < eps)[0]))

        cand_idxes = numpy.where(self.dist_matrix[idx] >= eps)[0]

        ncandis = len(cand_idxes)

        # it should have at least min_pts to be a cluster
        if ncandis >= min_pts:
            # check constraint among each candidates(name / year)
            idx_dist_tuples = list()
            for i in range(ncandis):
                idx_dist_tuples.append((cand_idxes[i], self.dist_matrix[idx, cand_idxes[i]]))

            #sorted_cand_idxes = sorted(cand_idxes, cmp=comp)
            sorted_cand_idxes = sorted(idx_dist_tuples, key=lambda tup: tup[1])

            #for cur_idx in sorted_cand_idxes:
            for cur_idx, cur_dist in sorted_cand_idxes:
                if cur_idx == idx:
                    continue

                for n_idx in neighbors:
                    #idx_check = AuthorBlock.get_idx_in_matrix(cur_idx, n_idx)
                    idx_check = (cur_idx, n_idx)
                    if self.dist_matrix[idx_check] <= 0.0:
                        break
                else:
                    neighbors.append(cur_idx)
        return neighbors

    @staticmethod
    def get_idx_in_matrix(source_idx, target_idx):
        ret_idx = (source_idx, target_idx) if target_idx > source_idx \
            else (target_idx, source_idx)
        return ret_idx

    @staticmethod
    def get_name_key(author):
        namekey = ""
        fname = author.get_first_name()
        if len(fname) > 0 and len(fname.strip()) > 0:
            namekey = fname
        return namekey

    @staticmethod
    def get_dist_key(aid1, aid2):
        if aid1 <= aid2:
            return str(aid1) + ":" + str(aid2)
        else:
            return str(aid2) + ":" + str(aid1)

    # old method to calculate neighbors
    '''
    """ Get neighbor aids of aid p1 within distance eps """
    def getNeighbors(self, p1, eps):
        neighbors = set()
        namekey = AuthorBlock.getNaxmeKey(self.points.get(p1).getAuthor())
        no_caches = set()

        # try if there is a cached result
        region0 = self.regions.get(namekey)

        if region0:
            self.getNeighborsByCache(p1, region0, eps, neighbors, no_caches)
        else:
            print ("Warning region0 is null")

        # if first name is full, check with the initial also
        if len(namekey) > 1:
            region1 = self.regions.get(namekey[:1])

            if region1:
                self.getNeighborsByCache(p1, region1, eps, neighbors, no_caches)

        # if there is a cache miss, get the result with pairwise classifier
        if len(no_caches):
            self.getNeighborsByService(p1, no_caches, eps, neighbors)
        return neighbors

    def calcDist(self, forest, X):
        check_is_fitted(forest, 'n_outputs_')

        # Check data
        X = check_array(X, dtype=DTYPE, accept_sparse="csr")

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_estimators(forest.n_estimators,
                                                        forest.n_jobs)

        # Parallel loop
        all_preds = Parallel(n_jobs=n_jobs, verbose=forest.verbose, backend="threading")(
            delayed(_parallel_helper)(e, 'predict', X, check_input=False)
            for e in forest.estimators_)

        modes, counts = mode(all_preds, axis=0)

        ratio = [x / float(100) for x in counts[0]]

        cnt = 0
        for count in counts[0]:
            if modes[0][cnt] == 1:
                ratio[cnt] = 1 - ratio[cnt]
            cnt += 1

        return ratio

    """ Get neighbors from cached result """
    def getNeighborsByCache(self, p1, region, eps, neighbors, no_caches):
        point1 = self.points.get(p1)
        for p2 in region:
            point2 = self.points.get(p2)
            if p1 == p2 or point1.getDoc().getId() == point2.getDoc().getId():
                continue
            if not AuthorBlock.passConstrains(point1, point2):
                continue

            distkey = AuthorBlock.getDistKey(p1, p2)
            d = self.distCache.get(distkey)

            if d is not None:
                if d < eps:
                    neighbors.add(p2)
            else:
                no_caches.add(p2)

    """ Get neighbors using pairwise classifier """
    def getNeighborsByService(self, p1, region, eps, neighbors):
        if self.rf is None:
            self.rf = joblib.load('model/model_python.rf')

        mat_feats = numpy.zeros(shape=(len(region), SimilarityProfile.NFEAT))

        i = 0
        point1 = self.points.get(p1)
        for p2 in region:
            point2 = self.points.get(p2)
            feats = SimilarityProfile.calcFeatureVector(point1, point2)
            mat_feats[i, :] = feats
            i += 1
        mat_dist = self.rf.predict_proba(mat_feats)
        dists = mat_dist[:, 0].tolist()
        #dists = self.calcDist(self.rf, mat_feats)

        i = 0
        for p2 in region:
            dist = dists[i]
            distkey = AuthorBlock.getDistKey(p1, p2)
            self.distCache[distkey] = dist
            if dist < eps:
                neighbors.add(p2)
            i += 1
    '''
