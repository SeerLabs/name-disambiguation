import math

import numpy
from scipy.spatial import distance

from util import config
from dao.author_point import AuthorPoint


class SimilarityProfileBoW(object):
    """
    Class to generate feature vector for pairwise classifier
    """
    # TODO: automatically calculate this
    NFEAT = 8

    title_idf, affil_idf, journal_idf, author_idf, \
    year_diff, abstract_idf, title_w2v, abstract_w2v = range(NFEAT)

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

        def calc_feat(feat1, feat2):
            if len(feat1) > 0 and len(feat2) > 0:
                intersect = feat1.viewkeys() & feat2.viewkeys()
                if len(intersect) > 0:
                    feat1_norm = normailze_vector(feat1)
                    feat2_norm = normailze_vector(feat2)
                    return sum([feat1_norm[k] * feat2_norm[k] for k in intersect])
                else:
                    return 0.0
            else:
                return 0.0

        # fetaures related to publication year
        def calc_year_feats(year1, year2):
            if year1 < 1800 or year2 < 1800:
                j_year_diff = 0.0
            else:
                j_year_diff = min(50.0, abs(year1 - year2))
            return j_year_diff

        feat_vector = numpy.zeros(SimilarityProfileBoW.NFEAT)
        d1 = p1.get_doc()
        a1 = p1.get_author()

        if isinstance(p2, AuthorPoint):
            d2 = p2.get_doc()
            a2 = p2.get_author()
            feat_vector[SimilarityProfileBoW.title_idf] = \
                calc_feat(d1.title_bow, d2.title_bow)
            feat_vector[SimilarityProfileBoW.affil_idf] = \
                calc_feat(a1.affil_bow, a2.affil_bow)
            feat_vector[SimilarityProfileBoW.journal_idf] = \
                calc_feat(d1.journal_bow, d2.journal_bow)
            feat_vector[SimilarityProfileBoW.abstract_idf] = \
                calc_feat(d1.abstract_bow, d2.abstract_bow)
            feat_vector[SimilarityProfileBoW.author_idf] = \
                calc_feat(d1.author_bow, d2.author_bow)
            feat_vector[SimilarityProfileBoW.year_diff] = \
                calc_year_feats(d1.get_year(), d2.get_year())

            if config.INCLUDE_W2V:
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

        else:
            # p2 is AuthorCluster
            feat_vector[SimilarityProfileBoW.title_idf] = \
                calc_feat(d1.title_bow, p2.title_bow)
            feat_vector[SimilarityProfileBoW.affil_idf] = \
                calc_feat(a1.affil_bow, p2.affil_bow)
            feat_vector[SimilarityProfileBoW.journal_idf] = \
                calc_feat(d1.journal_bow, p2.journal_bow)
            feat_vector[SimilarityProfileBoW.abstract_idf] = \
                calc_feat(d1.abstract_bow, p2.abstract_bow)
            feat_vector[SimilarityProfileBoW.author_idf] = \
                calc_feat(d1.author_bow, p2.coauthor_bow)
            feat_vector[SimilarityProfileBoW.year_diff] = \
                calc_year_feats(d1.get_year(), p2.year)

            if config.INCLUDE_W2V:
                if not d1.title_w2v.any() or not p2.title_w2v.any():
                    feat_vector[SimilarityProfileBoW.title_w2v] = 0.0
                else:
                    feat_vector[SimilarityProfileBoW.title_w2v] = \
                        1.0 - distance.cosine(d1.title_w2v, p2.title_w2v)

                if not d1.abstract_w2v.any() or not p2.abstract_w2v.any():
                    feat_vector[SimilarityProfileBoW.abstract_w2v] = 0.0
                else:
                    feat_vector[SimilarityProfileBoW.abstract_w2v] = \
                        1.0 - distance.cosine(d1.abstract_w2v, p2.abstract_w2v)

        # add compatability
        # if a1.is_compatible(a2):
        #     feat_vector[SimilarityProfileBoW.comp] = 1.0
        # else:
        #     feat_vector[SimilarityProfileBoW.comp] = 0.0

        return feat_vector

