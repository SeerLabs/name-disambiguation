import math

import numpy
from scipy.spatial import distance

from util import config


class SimilarityProfileBoW(object):
    """
    Class to generate feature vector for pairwise classifier
    """
    # TODO: automatically calculate this
    NFEAT = 11
    NPAPERS = 24358073

    # title_idf, affil_idf, grant_idf, journal_idf, chemical_idf, mesh_idf, \
    # author_idf, year_diff, abstract_idf, title_w2v, abstract_w2v, comp = range(NFEAT)
    title_idf, affil_idf, grant_idf, journal_idf, chemical_idf, mesh_idf, \
    author_idf, year_diff, abstract_idf, title_w2v, abstract_w2v = range(NFEAT)
    # author_idf, year_diff = range(NFEAT)

    @staticmethod
    def calc_feature_vector(p1, p2):
        # normailze vector
        def normailze_vector(vec):
            total = math.sqrt(sum([vec[term] ** 2 for term in vec]))
            if total > 0:
                output_dict = {k: v / total for k, v in vec.iteritems()}
                return output_dict
            else:
                return vec

        # features related to title
        def calc_title_feats(title1, title2, feat_vector):
            if len(title1)>0 and len(title2)>0:
                intersect = title1.viewkeys() & title2.viewkeys()
                if len(intersect) > 0:
                    title1_norm = normailze_vector(title1)
                    title2_norm = normailze_vector(title2)
                    feat_vector[SimilarityProfileBoW.title_idf] = \
                        sum([title1_norm[k] * title2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.title_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.title_idf] = 0.0

        # features related to affiliation
        def calc_affil_feats(affil1, affil2, feat_vector):
            if len(affil1)>0 and len(affil2)>0:
                intersect = affil1.viewkeys() & affil2.viewkeys()
                if len(intersect) > 0:
                    affil1_norm = normailze_vector(affil1)
                    affil2_norm = normailze_vector(affil2)
                    feat_vector[SimilarityProfileBoW.affil_idf] = \
                        sum([affil1_norm[k] * affil2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.affil_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.affil_idf] = 0.0

        def calc_grant_feats(grant1, grant2, feat_vector):
            if len(grant1)>0 and len(grant2)>0:
                intersect = grant1.viewkeys() & grant2.viewkeys()
                if len(intersect) > 0:
                    grant1_norm = normailze_vector(grant1)
                    grant2_norm = normailze_vector(grant2)
                    feat_vector[SimilarityProfileBoW.grant_idf] = \
                        sum([grant1_norm[k] * grant2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.grant_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.grant_idf] = 0.0

        def calc_journal_feats(journal1, journal2, feat_vector):
            if len(journal1) > 0 and len(journal2) > 0:
                intersect = journal1.viewkeys() & journal2.viewkeys()
                if len(intersect) > 0:
                    journal1_norm = normailze_vector(journal1)
                    journal2_norm = normailze_vector(journal2)
                    feat_vector[SimilarityProfileBoW.journal_idf] = \
                        sum([journal1_norm[k] * journal2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.journal_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.journal_idf] = 0.0

        def calc_abstract_feats(abstract1, abstract2, feat_vector):
            if len(abstract1) > 0 and len(abstract2) > 0:
                intersect = abstract1.viewkeys() & abstract2.viewkeys()
                if len(intersect) > 0:
                    abstract1_norm = normailze_vector(abstract1)
                    abstract2_norm = normailze_vector(abstract2)
                    feat_vector[SimilarityProfileBoW.abstract_idf] = \
                        sum([abstract1_norm[k] * abstract2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.abstract_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.abstract_idf] = 0.0

        def calc_chemical_feats(chemical1, chemical2, feat_vector):
            if len(chemical1) > 0 and len(chemical2) > 0:
                intersect = chemical1.viewkeys() & chemical2.viewkeys()
                if len(intersect) > 0:
                    chemical1_norm = normailze_vector(chemical1)
                    chemical2_norm = normailze_vector(chemical2)
                    feat_vector[SimilarityProfileBoW.chemical_idf] = \
                        sum([chemical1_norm[k] * chemical2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.chemical_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.chemical_idf] = 0.0

        # features related to MeSH
        def calc_mesh_feats(meshes1, meshes2, feat_vector):
            if len(meshes1) > 0 and len(meshes2) > 0:
                intersect = meshes1.viewkeys() & meshes2.viewkeys()
                if len(intersect) > 0:
                    meshes1_norm = normailze_vector(meshes1)
                    meshes2_norm = normailze_vector(meshes2)
                    feat_vector[SimilarityProfileBoW.mesh_idf] = \
                        sum([meshes1_norm[k] * meshes2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.mesh_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.mesh_idf] = 0.0

        # features realted to coauthors
        def calc_coauth_feats(coauthor1, coauthor2, feat_vector):
            # we have to remove the author itself
            # signature = p1.firstName[:1] + "_" + p1.lastName
            # coauth1 = copy.deepcopy(coauthor1)
            # coauth1.pop(signature, None)
            # coauth2 = copy.deepcopy(coauthor2)
            # coauth2.pop(signature, None)
            #
            # if len(coauth1) > 0:
            #     # normalize vector again
            #     total = math.sqrt(sum([coauth1[term] for term in coauth1]))
            #     coauth1 = {k: v / total for k, v in coauth1.iteritems()}
            # if len(coauth2) > 0:
            #     # normalize vector again
            #     total = math.sqrt(sum([coauth2[term] for term in coauth2]))
            #     coauth2 = {k: v / total for k, v in coauth2.iteritems()}

            if len(coauthor1)>0 and len(coauthor2)>0:
                intersect = coauthor1.viewkeys() & coauthor2.viewkeys()
                if len(intersect) > 0:
                    coauthor1_norm = normailze_vector(coauthor1)
                    coauthor2_norm = normailze_vector(coauthor2)
                    feat_vector[SimilarityProfileBoW.author_idf] = \
                        sum([coauthor1_norm[k] * coauthor2_norm[k] for k in intersect])
                else:
                    feat_vector[SimilarityProfileBoW.author_idf] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.author_idf] = 0.0

        # fetaures related to publication year
        def calc_year_feats(year1, year2, feat_vector):
            if year1 < 1800 or year2 < 1800:
                j_year_diff = 0.0
            else:
                j_year_diff = min(50.0, abs(year1 - year2))

            feat_vector[SimilarityProfileBoW.year_diff] = j_year_diff
            
        # features related to keywords
        # def calc_keyword_feats(doc1, doc2, featVector):
        #     shared_keywords = doc1.get_shared_keywords(doc2)
        #     m = float(len(shared_keywords))
        #     featVector.append(m)
        #
        #     sum_idf = 0.0
        #     for keyword in shared_keywords:
        #         idf = SimilarityProfile.keyword_dict.get(keyword)
        #         if not idf:
        #             idf = 0.0
        #         sum_idf += idf
        #     featVector.append(sum_idf)
        
        feat_vector = numpy.zeros(SimilarityProfileBoW.NFEAT)
        d1 = p1.get_doc()
        d2 = p2.get_doc()
        a1 = p1.get_author()
        a2 = p2.get_author()
        calc_title_feats(d1.title_bow, d2.title_bow, feat_vector)
        calc_affil_feats(a1.affil_bow, a2.affil_bow, feat_vector)
        calc_grant_feats(d1.grant_bow, d2.grant_bow, feat_vector)
        calc_journal_feats(d1.journal_bow, d2.journal_bow, feat_vector)
        calc_abstract_feats(d1.abstract_bow, d2.abstract_bow, feat_vector)
        calc_chemical_feats(d1.chemical_bow, d2.chemical_bow, feat_vector)
        calc_mesh_feats(d1.mesh_bow, d2.mesh_bow, feat_vector)
        calc_coauth_feats(d1.author_bow, d2.author_bow, feat_vector)
        calc_year_feats(d1.get_year(), d2.get_year(), feat_vector)

        if config.INCLUDE_W2V:
            # calc title average word2vec and calculate distance
            # title_w2v_1 = calc_average_word2vec(p1.title)
            # title_w2v_2 = calc_average_word2vec(p2.title)
            # calc abstract average word2vec and calculate distance
            # abstract_w2v_1 = calc_average_word2vec(p1.abstract)
            # abstract_w2v_2 = calc_average_word2vec(p2.abstract)

            if not d1.title_w2v.any() or not d2.title_w2v.any():
                feat_vector[SimilarityProfileBoW.title_w2v] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.title_w2v] = \
                    1.0 - distance.cosine(d1.title_w2v, d2.title_w2v)

            if not d1.abstract_w2v.any() or not d2.abstract_w2v.any():
                feat_vector[SimilarityProfileBoW.abstract_w2v] = 0.0
            else:
                feat_vector[SimilarityProfileBoW.abstract_w2v] = \
                    1.0 - distance.cosine(d1.abstract_w2v, d2.abstract_w2v)

        #add compatability
        # if a1.is_compatible(a2):
        #     feat_vector[SimilarityProfileBoW.comp] = 1.0
        # else:
        #     feat_vector[SimilarityProfileBoW.comp] = 0.0

        return feat_vector