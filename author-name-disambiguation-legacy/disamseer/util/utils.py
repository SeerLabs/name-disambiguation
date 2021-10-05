import cPickle as pickle
import math
from string import punctuation
import MySQLdb

import gensim
from nltk.corpus import stopwords

import config

#class Utils(object):
EPS = 2.2250738585072014e-308
STOP_WORDS = None
TITLE_IDF_DICT = None
AFFIL_IDF_DICT = None
GRANT_IDF_DICT = None
JOURNAL_IDF_DICT = None
ABSTRACT_IDF_DICT = None
CHEMICAL_IDF_DICT = None
MESH_IDF_DICT = None
NAME_IDF_DICT = None

FILTER_LOW_VAL = False
NDOCS = 24358073.0
THRESHOLD_SMALL = 5
THRESHOLD_LARGE = 10
THRESHOLD_MAX = 20

print 'Load stopwords..'
STOP_WORDS = set(stopwords.words('english'))
# add pubmed stopwords
with open('resources/pubmed_stopwords.txt') as fp:
    for line in fp:
        STOP_WORDS.add(line.strip().lower())

#def load_idf_dicts():

def filter_dict(input_dict, threshold):
    for key in input_dict.keys():
        cur_val = (1.0 / math.exp(input_dict[key])) * NDOCS
        if cur_val <= threshold:
            del input_dict[key]
    return input_dict

if not TITLE_IDF_DICT:
    print 'Load idf files..'
    with open(config.TITLE_IDF_FILE, 'rb') as fp:
        TITLE_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            TITLE_IDF_DICT = filter_dict(TITLE_IDF_DICT,THRESHOLD_SMALL)
    # title_vectorizer = CountVectorizer(vocabulary=TITLE_IDF_DICT.keys(), dtype=numpy.int32)
    # title_vectorizer._validate_vocabulary()
    # print 'title'
    # print len(title_vectorizer.get_feature_names())

    with open(config.AFFIL_IDF_FILE, 'rb') as fp:
        AFFIL_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            AFFIL_IDF_DICT = filter_dict(AFFIL_IDF_DICT,THRESHOLD_MAX)
    # affil_vectorizer = CountVectorizer(vocabulary=AFFIL_IDF_DICT.keys(), dtype=numpy.int32)
    # affil_vectorizer._validate_vocabulary()
    # print 'affil'
    # print len(affil_vectorizer.get_feature_names())

    with open(config.GRANT_IDF_FILE, 'rb') as fp:
        GRANT_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            GRANT_IDF_DICT = filter_dict(GRANT_IDF_DICT,THRESHOLD_SMALL)
    GRANT_DICT = dict()
    cnt = 0
    for grant in GRANT_IDF_DICT.keys():
        GRANT_DICT[grant] = cnt
        cnt += 1

    with open(config.JOURNAL_IDF_FILE, 'rb') as fp:
        JOURNAL_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            JOURNAL_IDF_DICT = filter_dict(JOURNAL_IDF_DICT,THRESHOLD_SMALL)
    # journal_vectorizer = CountVectorizer(vocabulary=JOURNAL_IDF_DICT.keys(), dtype=numpy.int32)
    # journal_vectorizer._validate_vocabulary()
    # print 'journal'
    # print len(journal_vectorizer.get_feature_names())

    with open(config.ABSTRACT_IDF_FILE, 'rb') as fp:
        ABSTRACT_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            ABSTRACT_IDF_DICT = filter_dict(ABSTRACT_IDF_DICT,THRESHOLD_LARGE)
    # abstract_vectorizer = CountVectorizer(vocabulary=ABSTRACT_IDF_DICT.keys(), dtype=numpy.int32)
    # abstract_vectorizer._validate_vocabulary()
    # print 'abstract'
    # print len(abstract_vectorizer.get_feature_names())

    with open(config.CHEMICAL_IDF_FILE, 'rb') as fp:
        CHEMICAL_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            CHEMICAL_IDF_DICT = filter_dict(CHEMICAL_IDF_DICT,THRESHOLD_SMALL)
    # CHEMICAL_DICT = dict()
    # cnt = 0
    # for chem in CHEMICAL_IDF_DICT.keys():
    #     CHEMICAL_DICT[chem] = cnt
    #     cnt += 1

    with open(config.MESH_IDF_FILE, 'rb') as fp:
        MESH_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            MESH_IDF_DICT = filter_dict(MESH_IDF_DICT,THRESHOLD_SMALL)
    # MESH_DICT = dict()
    # cnt = 0
    # for mesh in MESH_IDF_DICT.keys():
    #     MESH_DICT[mesh] = cnt
    #     cnt += 1

    with open(config.NAME_IDF_FILE, 'rb') as fp:
        NAME_IDF_DICT = pickle.load(fp)
        if FILTER_LOW_VAL:
            NAME_IDF_DICT = filter_dict(NAME_IDF_DICT,THRESHOLD_LARGE)
    # NAME_DICT = dict()
    # cnt = 0
    # for name in NAME_IDF_DICT.keys():
    #     NAME_DICT[name] = cnt
    #     cnt += 1
    print 'Load idf files ended..'

if config.INCLUDE_W2V:
    W2V_MODEL = gensim.models.Word2Vec.load('resources/title_abstract.w2v')
    print 'Load W2V model ended..'


def remove_punc(s):
    return s.encode('utf-8').translate(None, punctuation).decode('utf-8')


def connect_db():
    db = MySQLdb.connect(config.DB_HOST, config.DB_USER, config.DB_PWD,
                         config.DB_NAME, charset='utf8', use_unicode=True,
                         port=config.DB_PORT)
    return db