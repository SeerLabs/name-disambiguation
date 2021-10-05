import numpy

from author_block import AuthorPoint
from parse_error import NameParseError
from pubmed_doc import PubmedDoc
from util import config, utils


class AuthorCluster(object):
    """
    Class that holds all author mentions of each disambiguated cluster
    """
    def __init__(self, cursor, aids):
        self.points = list()

        self.title_bow = dict()
        self.abstract_bow = dict()
        self.coauthor_bow = dict()
        self.affil_bow = dict()
        self.journal_bow = dict()

        self.var_names = set()
        self.year = 0

        self.title_w2v = None
        self.abstract_w2v = None

        if isinstance(aids, list):
            for aid in aids:
                try:
                    doc = PubmedDoc(cursor, aid, False)
                except NameParseError as e:
                    print e
                    continue
                auth = doc.get_author()
                point = AuthorPoint(aid, auth, doc)
                self.points.append(point)

        # calculate universal BoW vectors
        self.calc_cluster_bow()
        self.find_var_names()

    def calc_cluster_bow(self):
        def add_dict(target_dict, input_dict):
            for key in input_dict:
                if target_dict.get(key) is None:
                    target_dict[key] = input_dict.get(key)
                else:
                    target_dict[key] += input_dict.get(key)

        # def normalize_dict(target_dict):
        #     total = math.sqrt(sum([target_dict[term]**2 for term in target_dict]))
        #     if total > 0:
        #         normalized_dict = {k: v/total for k, v in target_dict.iteritems()}
        #         return normalized_dict
        #     else:
        #         return target_dict

        year_count = 0
        for point in self.points:
            doc = point.get_doc()
            # title
            add_dict(self.title_bow, doc.title_bow)
            # abstract
            add_dict(self.abstract_bow, doc.abstract_bow)
            # coauthor
            add_dict(self.coauthor_bow, doc.author_bow)
            # affil
            add_dict(self.affil_bow, doc.get_author().affil_bow)
            # journal
            add_dict(self.journal_bow, doc.journal_bow)

            if doc.year > 0:
                self.year += doc.year
                year_count += 1

        # self.title_bow = normalize_dict(self.title_bow)
        # self.abstract_bow = normalize_dict(self.abstract_bow)
        # self.coauthor_bow = normalize_dict(self.coauthor_bow)
        # self.affil_bow = normalize_dict(self.affil_bow)
        # self.journal_bow = normalize_dict(self.journal_bow)

        if year_count > 0:
            self.year /= year_count

        if config.INCLUDE_W2V:
            self.title_w2v = numpy.zeros(300)
            self.abstract_w2v = numpy.zeros(300)

            for token in self.title_bow:
                if token in utils.W2V_MODEL.wv.vocab:
                    self.title_w2v += self.title_bow[token] * utils.W2V_MODEL[token]

            for token in self.abstract_bow:
                if token in utils.W2V_MODEL.wv.vocab:
                    self.abstract_w2v += self.abstract_bow[token] * utils.W2V_MODEL[token]

    # find all variation of the name
    def find_var_names(self):
        for point in self.points:
            doc = point.get_doc()
            self.var_names.add(doc.get_author().get_first_middle())