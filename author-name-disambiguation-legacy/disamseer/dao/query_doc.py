from citation import Citation
from util import config, utils
from query_author import QueryAuthor
from parse_error import NameParseError

import numpy


class QueryDoc(Citation):

    def __init__(self, input, order=1):
        self.author_names = list()
        self.abstract = ""
        self.author = None
        self.venue = ""

        # BoW vectors
        self.title_bow = dict()
        self.abstract_bow = dict()
        self.author_bow = dict()
        self.journal_bow = dict()

        self.title_w2v = None
        self.abstract_w2v = None

        if input.get('author') is not None:
            for auth in input.get('author'):
                try:
                    author = QueryAuthor(auth)
                except NameParseError as e:
                    print "PubmedDoc : " + str(e)
                    # if isDoc or (author is not None and author.get_id() != inputid):
                    #     continue
                    # else:
                    #     raise NameParseError("PubmedDoc cannot be generated")
                    #     break
                    continue
                signature = author.firstName.strip().lower()[:1] + "_" + \
                            author.lastName.strip().lower()

                if author.order != order:
                    self.author_names.append(signature)
                else:
                    self.author = author

            if self.author is None:
                raise NameParseError("PubmedDoc cannot be generated")

            #self.author = QueryAuthor(input.authors[order-1])

        if input.get('abstract') is not None:
            self.abstract = input.get('author').lower().strip()

        if input.get('title') is not None:
            self.title = input.get('title').lower().strip()

        if input.get('booktitle') is not None:
            self.venue = input.get('booktitle').lower().strip()
        elif input.get('journal') is not None:
            self.venue = input.get('journal').lower().strip()
        else:
            self.venue = ""

        if input.get('year') is not None and input.get('year') > 500:
            self.year = int(input.get('year'))

        if config.CALC_BOW:
            self.calc_bow_vectors()

    def calc_bow_vectors(self):
        def calc_bow_vector(input_list, idf_dict):
            output_dict = {}
            for item in input_list:
                weight = idf_dict.get(item)
                if weight is None:
                    weight = 0.0
                    # print item + ' no weight'
                if output_dict.get(item) is None:
                    output_dict[item] = weight
                else:
                    output_dict[item] += weight

            # normalize when we calculate the features.. (in SimilarityProfile)
            # total = math.sqrt(sum([output_dict[term]**2 for term in output_dict]))
            # if total > 0:
            #     output_dict = {k: v/total for k,v in output_dict.iteritems()}

            return output_dict

        # calculate title BoW vector
        if len(self.title) > 0:
            #print utils.STOP_WORDS
            #print utils.TITLE_IDF_DICT
            title_cleaned = utils.remove_punc(self.title)
            title_list = [token for token in title_cleaned.split() if token not in utils.STOP_WORDS]
            self.title_bow = calc_bow_vector(title_list, utils.TITLE_IDF_DICT)

        # calculate journal BoW vector
        if len(self.venue) > 0:
            venue_cleaned = utils.remove_punc(self.venue)
            venue_list = [token for token in venue_cleaned.split() if token not in utils.STOP_WORDS]
            self.journal_bow = calc_bow_vector(venue_list, utils.JOURNAL_IDF_DICT)

        # calculate abstract BoW vector
        if len(self.abstract) > 0:
            abstract_cleaned = utils.remove_punc(self.abstract)
            abstract_list = [token for token in abstract_cleaned.split() if token not in utils.STOP_WORDS]
            self.abstract_bow = calc_bow_vector(abstract_list, utils.ABSTRACT_IDF_DICT)

        # calculate author BoW vector
        if len(self.author_names) > 0:
            self.author_bow = calc_bow_vector(self.author_names, utils.NAME_IDF_DICT)

        if config.INCLUDE_W2V:
            self.title_w2v = numpy.zeros(300)
            self.abstract_w2v = numpy.zeros(300)

            for token in self.title_bow:
                if token in utils.W2V_MODEL.wv.vocab:
                    self.title_w2v += self.title_bow[token] * utils.W2V_MODEL[token]

            for token in self.abstract_bow:
                if token in utils.W2V_MODEL.wv.vocab:
                    self.abstract_w2v += self.abstract_bow[token] * utils.W2V_MODEL[token]

    def get_author(self):
        return self.author
    def get_abstract(self):
        return self.abstract

    def set_abstract(self, abstract):
        self.abstract = abstract
