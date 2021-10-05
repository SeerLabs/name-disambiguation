from string import punctuation

import jellyfish
import numpy
from scipy.spatial import distance

from util import config, utils


class SimilarityProfile(object):
    """
    Class to generate feature vector for pairwise classifier
    """
    # TODO: automatically calculate this
    NFEAT = 14
    NPAPERS = 24358073
    EPS = 2.2250738585072014e-308

    auth_fst, auth_mid, auth_ord, year, year_diff, \
    mesh_shared, mesh_shared_idf, title_shared, aff_jac, aff_jw, \
    coauth_shared, coauth_idf, coauth_jac, jour_shared_idf, \
    = range(NFEAT)

    '''
    print 'Load stopwords..'
    stop = set(stopwords.words('english'))
    print 'Load idf files..'
    lname_dict = dict()
    journal_dict = dict()
    lang_dict = dict()
    mesh_dict = dict()
    # keyword_dict = dict()
    with codecs.open(config.DF_LASTNAME_FILE, 'r', encoding='utf8') as fp:
        for line in fp:
            name = line.split('\t')
            lname_dict[name[0].strip()] = math.log(NPAPERS / float(name[1]))
    with codecs.open(config.DF_JOURNAL_FILE, 'r', encoding='utf8') as fp:
        for line in fp:
            journal = line.split('\t')
            journal_dict[journal[0].strip()] = math.log(NPAPERS / float(journal[1]))
    with codecs.open(config.DF_LANGUAGE_FILE, 'r', encoding='utf8') as fp:
        for line in fp:
            lang = line.split('\t')
            lang_dict[lang[0].strip()] = math.log(NPAPERS / float(lang[1]))
    with codecs.open(config.DF_MESH_FILE, 'r', encoding='utf8') as fp:
        for line in fp:
            mesh = line.split('\t')
            mesh_dict[mesh[0].strip()] = math.log(NPAPERS / float(mesh[1]))
    with codecs.open(config.DF_KEYWORD_FILE, 'r', encoding='utf8') as fp:
        for line in fp:
            keyword = line.split('\t')
            print keyword[0]
            keyword_dict[keyword[0].strip()] = math.log(NPAPERS / float(keyword[1]))
    print 'Load idf files ended..'
    '''

    @staticmethod
    def calc_feature_vector(p1, p2):
        # features related to author name and order
        def calc_author_feats(p1, p2, feat_vector):
            auth1 = p1.get_author()
            auth2 = p2.get_author()

            """ first name """
            f1 = auth1.get_first_name().lower()
            f2 = auth2.get_first_name().lower()

            f = 0.0
            f_jw = 0.0

            if len(f1) == 0 or len(f2) == 0:
                f = 1.0
            elif len(f1) > 1 and len(f2) > 1:
                if f1 == f2:
                    f = 3.0
                else:
                    f = 0.0
            else:
                if f1[:1] == f2[:1]:
                    f = 2.0
                else:
                    f = 1.0
            feat_vector[SimilarityProfile.auth_fst] = f
            # featVector.append(f)

            """ first name jaro-winkler """
            '''
            f_jw = jellyfish.jaro_winkler(unicode(f1), unicode(f2))
            if f_jw < 0.0:
                f_jw = 0.0
            featVector.append(f_jw)
            '''

            """ middle name """
            m1 = auth1.get_middle_name().lower()
            m2 = auth2.get_middle_name().lower()

            m = 0.0
            if len(m1) == 1 and len(m2) > 1:
                m2 = m2[:1]
            elif len(m2) == 1 and len(m1) > 1:
                m1 = m1[:1]

            if m1 == m2:
                if len(m1) > 0:
                    m = 3.0
                else:
                    m = 2.0
            else:
                if len(m1) > 0 and len(m2) > 0:
                    m = 0.0
                else:
                    m = 1.0
            # featVector.append(m)
            feat_vector[SimilarityProfile.auth_mid]= m

            # middle name jaro-winkler
            # m_jw = jellyfish.jaro_winkler(unicode(m1), unicode(m2))
            # if m_jw < 0.0:
            #     m_jw = 0.0
            # featVector.append(m_jw)

            """ last name IDF """
            # lname_idf = SimilarityProfile.lname_dict.get(auth1.get_last_name().lower().strip())
            # if not lname_idf:
            #     lname_idf = 0.0
            # # featVector.append(lname_idf)
            # feat_vector[SimilarityProfile.auth_last_idf] = lname_idf

            """ author order """
            o = 0.0
            if auth1.get_order() == 1 and auth2.get_order() == 1:
                o = 2.0
            elif auth1.get_order() == p1.get_doc().get_num_authors() and \
                auth2.get_order() == p2.get_doc().get_num_authors():
                o = 1.0
            else:
                o = 0.0
            # featVector.append(o)
            feat_vector[SimilarityProfile.auth_ord] = o

        # fetaures related to publication year
        def calc_year_feats(year1, year2, feat_vector):
            if year1 < 1800 or year2 < 1800:
                j_year = 0.0
                j_year_diff = 0.0
            else:
                if year1 < 1988 and year2 < 1988:
                    j_year = 0.0
                elif (year1 >= 1988 and year2 < 1988) or (year1 < 1988 and year2 >= 1988):
                    j_year = 1.0
                elif year1 >= 1988 and year2 >= 1988 and year1 <= 2002 and year2 <= 2002:
                    j_year = 2.0
                elif year1 > 2002 and year2 > 2002:
                    j_year = 4.0
                else:
                    j_year = 3.0

                j_year_diff = abs(year1 - year2)

            #featVector.append(j_year)
            #featVector.append(float(j_year_diff))
            feat_vector[SimilarityProfile.year] = j_year
            feat_vector[SimilarityProfile.year_diff] = j_year_diff

        # features related to MeSH
        def calc_mesh_feats(meshes1, meshes2, feat_vector):
            shared_meshes = []
            m = 0.0
            if not meshes1 and not meshes2:
                m = 0.0
            else:
                for mesh1 in meshes1:
                    for mesh2 in meshes2:
                        if mesh1.lower() == mesh2.lower():
                            shared_meshes.append(mesh1.lower())
                m = float(len(shared_meshes))
            #featVector.append(m)
            feat_vector[SimilarityProfile.mesh_shared] = m

            sum_idf = 0.0
            for mesh in shared_meshes:
                idf = utils.MESH_IDF_DICT.get(mesh)
                if not idf:
                    idf = 0.0
                sum_idf += idf
            #featVector.append(sum_idf)
            feat_vector[SimilarityProfile.mesh_shared_idf] = sum_idf

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

        # features related to title
        def calc_title_feats(title1, title2, feat_vector):
            if not title1 and not title2:
                t = 0.0
            else:
                # remove punctuation, probably move it to PubmedDoc?
                title1_remove_punc = ''.join(c for c in title1.lower() if c not in punctuation)
                title2_remove_punc = ''.join(c for c in title2.lower() if c not in punctuation)

                # filter stop words
                title1_filtered = [i for i in title1_remove_punc.split() if i not in utils.STOP_WORDS]
                title2_filtered = [i for i in title2_remove_punc.split() if i not in utils.STOP_WORDS]

                t = compute_jaccard_index(set(title1_filtered), set(title2_filtered))
                if t < 0.0:
                    t = 0.0
            #featVector.append(t)
            feat_vector[SimilarityProfile.title_shared] = t

        # features related to affiliation
        def calc_affil_feats(affil1, affil2, feat_vector):
            affil1_remove_punc = ''.join(c for c in affil1.lower() if c not in punctuation)
            affil2_remove_punc = ''.join(c for c in affil2.lower() if c not in punctuation)

            affil1_filtered = [i for i in affil1_remove_punc.split() if i not in utils.STOP_WORDS]
            affil2_filtered = [i for i in affil2_remove_punc.split() if i not in utils.STOP_WORDS]

            affil1_string = ""
            for word in affil1_filtered:
                affil1_string += word + " "
            if len(affil1_string) > 0:
                affil1_string = affil1_string[:-1]
            affil2_string = ""
            for word in affil2_filtered:
                affil2_string += word + " "
            if len(affil2_string) > 0:
                affil2_string = affil2_string[:-1]

            if not affil1_remove_punc or not affil2_remove_punc:
                a_jac = 0.0
                a_jw = 0.0
            else:
                a_jac = compute_jaccard_index(set(affil1_filtered),
                                              set(affil2_filtered))
                if a_jac < 0.0:
                    a_jac = 0.0
                a_jw = jellyfish.jaro_winkler(unicode(affil1_string),
                                              unicode(affil2_string))
            # featVector.append(a_jac)
            # featVector.append(a_jw)
            feat_vector[SimilarityProfile.aff_jac] = a_jac
            feat_vector[SimilarityProfile.aff_jw] = a_jw

            # TODO: affiliation TFIDF / SOFT TFIDF

        # features realted to coauthors
        def calc_coauth_feats(p1, p2, feat_vector):
            shared_authors = p1.get_doc().get_shared_authors(p2.get_doc(), p1.get_author().get_last_name().lower().strip())
            """ number of shared coauthors """
            #featVector.append(float(len(shared_authors)))
            feat_vector[SimilarityProfile.coauth_shared] = float(len(shared_authors))

            """ sum of shared lastname IDF """
            coauthor1 = p1.get_doc().author_bow
            coauthor2 = p2.get_doc().author_bow
            if len(coauthor1) > 0 and len(coauthor2) > 0:
                intersect = coauthor1.viewkeys() & coauthor2.viewkeys()
                feat_vector[SimilarityProfile.coauth_idf] = \
                    sum([coauthor1[k] * coauthor2[k] for k in intersect])
            else:
                feat_vector[SimilarityProfile.coauth_idf] = 0.0

            # score_sum = 0.0
            # for lname in shared_authors:
            #     score = SimilarityProfile.lname_dict.get(lname)
            #     if not score:
            #         score = 0.0
            #     score_sum += score
            # #featVector.append(score_sum)
            # feat_vector[SimilarityProfile.coauth_idf] = score_sum

            """ jaccard distance of coauthors """
            coauth_jac = float(len(shared_authors)) / (p1.get_doc().get_num_authors()
                                                       + p2.get_doc().get_num_authors()
                                                       - len(shared_authors) + SimilarityProfile.EPS)
            #featVector.append(coauth_jac)
            feat_vector[SimilarityProfile.coauth_jac] = coauth_jac

        def calc_jour_feats(doc1, doc2, feat_vector):
            # if doc1.get_venue() == doc2.get_venue():
            #     journal_idf = SimilarityProfile.journal_dict.get(doc1.get_venue())
            #     if not journal_idf:
            #         journal_idf = 0.0
            #     #featVector.append(journal_idf)
            #     feat_vector[SimilarityProfile.jour_shared_idf] = journal_idf
            # else:
            #     #featVector.append(0.0)
            #     pass

            journal1 = doc1.journal_bow
            journal2 = doc2.journal_bow

            if len(journal1) > 0 and len(journal2) > 0:
                intersect = journal1.viewkeys() & journal2.viewkeys()
                feat_vector[SimilarityProfile.jour_shared_idf] = \
                    sum([journal1[k] * journal2[k] for k in intersect])
            else:
                feat_vector[SimilarityProfile.jour_shared_idf] = 0.0
            
            '''
            """ journal language features """
            lang_feat = 0.0
            lang_idf_feat = 0.0
            if doc1.getLanguage() == doc2.getLanguage():
                if not len(doc1.getLanguage()):
                    lang_feat = 0.0
                elif doc1.getLanguage() == 'eng':
                    lang_feat = 2.0
                else:
                    lang_feat = 3.0
            else:
                if not len(doc1.getLanguage()) or not len(doc2.getLanguage()):
                    lang_feat = 0.0
                elif doc1.getLanguage() == 'eng' or doc2.getLanguage() == 'eng':
                    lang_feat = 1.0
                else:
                    lang_feat = 0.0
            featVector.append(lang_feat)

            if not len(doc1.getLanguage()) or not len(doc2.getLanguage()):
                lang_idf_feat = 0.0
            else:
                if doc1.getLanguage() == doc2.getLanguage():
                    lang_idf_feat = SimilarityProfile.lang_dict.get(doc1.getLanguage())
                    if not lang_idf_feat:
                        lang_idf_feat = 0.0
                else:
                    lang_idf_feat = 0.0
            featVector.append(lang_idf_feat)
            '''

        def compute_jaccard_index(set1, set2):
            n = len(set1.intersection(set2))
            return n / float(len(set1) + len(set2) - n + SimilarityProfile.EPS)

        #featVector = list()
        feat_idx = 0
        feat_vector = numpy.zeros(SimilarityProfile.NFEAT)
        calc_author_feats(p1, p2, feat_vector)
        calc_year_feats(p1.get_doc().get_year(), p2.get_doc().get_year(), feat_vector)
        calc_mesh_feats(p1.get_doc().get_meshes(), p2.get_doc().get_meshes(), feat_vector)
        calc_title_feats(p1.get_doc().get_title(), p2.get_doc().get_title(), feat_vector)
        calc_affil_feats(p1.get_author().get_affil(), p2.get_author().get_affil(), feat_vector)
        calc_coauth_feats(p1, p2, feat_vector)
        calc_jour_feats(p1.get_doc(), p2.get_doc(), feat_vector)
        # calc_keyword_feats(p1.getDoc(), p2.getDoc(), featVector)
        # if p1.get_author().is_compatible(p2.get_author()):
        #     feat_vector[SimilarityProfile.comp] = 1.0
        # else:
        #     feat_vector[SimilarityProfile.comp] = 0.0

        if config.INCLUDE_W2V:
            # calc title average word2vec and calculate distance
            # title_w2v_1 = calc_average_word2vec(p1.title)
            # title_w2v_2 = calc_average_word2vec(p2.title)
            # calc abstract average word2vec and calculate distance
            # abstract_w2v_1 = calc_average_word2vec(p1.abstract)
            # abstract_w2v_2 = calc_average_word2vec(p2.abstract)

            d1 = p1.get_doc()
            d2 = p2.get_doc()

            if not d1.title_w2v.any() or not d2.title_w2v.any():
                feat_vector[SimilarityProfile.title_w2v] = 0.0
            else:
                feat_vector[SimilarityProfile.title_w2v] = \
                    1.0 - distance.cosine(d1.title_w2v, d2.title_w2v)

            if not d1.abstract_w2v.any() or not d2.abstract_w2v.any():
                feat_vector[SimilarityProfile.abstract_w2v] = 0.0
            else:
                feat_vector[SimilarityProfile.abstract_w2v] = \
                    1.0 - distance.cosine(d1.abstract_w2v, d2.abstract_w2v)

        return feat_vector






















