"""
Contains metadata for each Pubmed document
inherited Citation class
"""
import numpy

from citation import Citation
from util import config, utils
from pubmed_author import PubmedAuthor
from parse_error import NameParseError


class PubmedDoc(Citation):

    query = "SELECT pmid, title, year(date) as year, language, venue, abstract," \
            "journalid FROM papers WHERE pmid = %s"
    query_aid = "SELECT pmid, title, year(date) as year, language, venue, abstract, " \
                "journalid FROM papers WHERE pmid = (SELECT paperid FROM authors WHERE id = %s)"

    auth_query = "SELECT id, firstname, lastname, affiliation, ord FROM authors WHERE paperid = %s"
    mesh_query = "SELECT descriptor FROM meshes WHERE paperid = %s"
    keyword_query = "SELECT keyword FROM keywords WHERE paperid = %s"
    chemical_query = "SELECT ui FROM chemicals WHERE paperid = %s"
    grant_query = "SELECT grantid FROM grants WHERE paperid = %s"

    def __init__(self, cursor=None, inputid=None, isDoc=None):
        super(PubmedDoc, self).__init__()

        self.meshes = []
        self.chemicals = []
        self.grants = []
        self.author_names = []
        self.abstract = ""
        self.language = ""
        self.author = None

        # BoW vectors
        self.mesh_bow = dict()
        self.chemical_bow = dict()
        self.grant_bow = dict()
        self.title_bow = dict()
        self.abstract_bow = dict()
        self.author_bow = dict()
        self.journal_bow = dict()

        self.title_w2v = None
        self.abstract_w2v = None

        if cursor is not None:
            if isDoc:
                cursor.execute(PubmedDoc.query % inputid)

            else:
                cursor.execute(PubmedDoc.query_aid % inputid)

            result = cursor.fetchone()
            if result is not None:
                self.id = result[0]
                self.title = result[1].lower().strip()
                self.year = result[2]
                self.language = result[3].lower().strip()
                if not self.language:
                    self.language = ""
                self.venue = result[4].lower().strip()
                if not self.venue:
                    self.venue = ""
                self.abstract = result[5].lower().strip()
                if not self.abstract:
                    self.abstract = ""
                self.journalid = result[6].lower().strip()
                if not self.journalid:
                    self.journalid = ""

                cursor.execute(PubmedDoc.auth_query % self.id)
                auths = cursor.fetchall()
                for auth in auths:
                    try:
                        author = PubmedAuthor(auth)
                    except NameParseError as e:
                        print "PubmedDoc : " + str(e)
                        # if isDoc or (author is not None and author.get_id() != inputid):
                        #     continue
                        # else:
                        #     raise NameParseError("PubmedDoc cannot be generated")
                        #     break
                        continue
                    signature = author.firstName.strip().lower()[:1] + "_" +\
                                author.lastName.strip().lower()

                    if isDoc or author.get_id() != inputid:
                        self.author_names.append(signature)
                    else:
                        self.author = author
                    self.authors.append(author)

                if len(self.authors) == 0 or self.author is None:
                    raise NameParseError("PubmedDoc cannot be generated")

                cursor.execute(PubmedDoc.mesh_query % self.id)
                ms = cursor.fetchall()
                for mesh in ms:
                    self.meshes.append(mesh[0].lower().strip())

                cursor.execute(PubmedDoc.chemical_query % self.id)
                ch = cursor.fetchall()
                for chem in ch:
                    self.chemicals.append(chem[0].lower().strip())

                cursor.execute(PubmedDoc.grant_query % self.id)
                gr = cursor.fetchall()
                for grant in gr:
                    self.grants.append(grant[0].lower().strip())

                """
                cursor.execute(PubmedDoc.keyword_query % self.id)
                ks = cursor.fetchall()
                for keyword in ks:
                    cur_keyword = keyword[0].lower().replace('\n', ' ').replace('\t', ' ').strip()
                    self.keywords.add(cur_keyword)
                """
                
                if config.CALC_BOW:
                    self.calc_bow_vectors()

                # print '-----title-----'
                # print self.title
                # print self.title_bow
                # print '-----grant-----'
                # print self.grants
                # print self.grant_bow
                # print '-----journal---'
                # print self.venue
                # print self.journal_bow
                # print '-----chemical----'
                # print self.chemicals
                # print self.chemical_bow
                # print '-----mesh-----'
                # print self.meshes
                # print self.mesh_bow
                # print '----author-----'
                # print self.author_names
                # print self.author_bow
                # print '----abstract---'
                # print self.abstract
                # print self.abstract_bow
                # print '-------------------'

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

        # calculate grant BoW vector
        if len(self.grants) > 0:
            self.grant_bow = calc_bow_vector(self.grants, utils.GRANT_IDF_DICT)

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

        # calculate chemical BoW vector
        if len(self.chemicals) > 0:
            self.chemical_bow = calc_bow_vector(self.chemicals, utils.CHEMICAL_IDF_DICT)

        # calculate MeSH BoW vector
        if len(self.meshes) > 0:
            self.mesh_bow = calc_bow_vector(self.meshes, utils.MESH_IDF_DICT)

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

    def get_meshes(self):
        return self.meshes

    def get_chemicals(self):
        return self.chemicals

    def get_grants(self):
        return self.grants

    def get_abstract(self):
        return self.abstract

    def get_language(self):
        return self.language

    def get_journalid(self):
        return self.journalid

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_chemical(self, chem):
        self.chemicals.append(chem)

    def add_grant(self, grant):
        self.grants.append(grant)

    def set_language(self, lang):
        self.language = lang

    def set_abstract(self, abstract):
        self.abstract = abstract

    def set_journalid(self, journalid):
        self.journalid = journalid
