import os.path
import sys

import pandas as pd
import numpy as np

from src.disamseer.author_block import AuthorBlock

COLS = ['aid', 'label',
        'fn_f', 'fn_1', 'fn_3', 'fn_5', 'fn_7', 'fn_b1', 'fn_b3', 'fn_b5', 'fn_b7',
        'ln_f', 'ln_1', 'ln_3', 'ln_5', 'ln_7', 'ln_b1', 'ln_b3', 'ln_b5', 'ln_b7',
        'mn_f', 'mn_1', 'mn_3', 'mn_5', 'mn_7', 'mn_b1', 'mn_b3', 'mn_b5', 'mn_b7',
        'affil',
        'order',
        'year_f', 'year_1', 'year_2', 'year_3',
        'journalid']

COLS_TO_USE = ['fn_f', 'fn_1', 'fn_3', 'fn_5', 'fn_7', 'fn_b1', 'fn_b3', 'fn_b5', 'fn_b7',
               'ln_f', 'ln_1', 'ln_3', 'ln_5', 'ln_7', 'ln_b1', 'ln_b3', 'ln_b5', 'ln_b7',
               'mn_f', 'mn_1', 'mn_3', 'mn_5', 'mn_7', 'mn_b1', 'mn_b3', 'mn_b5', 'mn_b7',
               'affil',
               'order',
               'year_f',
               'year_1', 'year_2', 'year_3',
               'journalid']

COLS_TO_USE_APPENDED = ['fn_f', 'fn_1', 'fn_3', 'fn_5', 'fn_7', 'fn_b1', 'fn_b3', 'fn_b5', 'fn_b7', 'fn_c',
                        'ln_f', 'ln_1', 'ln_3', 'ln_5', 'ln_7', 'ln_b1', 'ln_b3', 'ln_b5', 'ln_b7', 'ln_c',
                        'mn_f', 'mn_1', 'mn_3', 'mn_5', 'mn_7', 'mn_b1', 'mn_b3', 'mn_b5', 'mn_b7', 'mn_c',
                        'title_cos_0.2', 'title_cos_0.4', 'title_cos_0.6', 'title_cos_0.8',
                        'affil', 'affil_cos_0.2', 'affil_cos_0.4', 'affil_cos_0.6', 'affil_cos_0.8', 'affil_c',
                        'coauth_cos_0.2', 'coauth_cos_0.4', 'coauth_cos_0.6', 'coauth_cos_0.8',
                        'order',
                        'year_f', 'year_1', 'year_2', 'year_3', 'year_c', 'year_diff_2', 'year_diff_5', 'year_diff_10',
                        'journalid', 'venue_cos_0.2', 'venue_cos_0.4', 'venue_cos_0.6', 'venue_cos_0.8']

def load_labels(label_path):
    def cos_dist(a, b):
        sim = sum([a[k] * b[k] for k in a.viewkeys() & b.viewkeys()])
        # print sim
        return 1.0 - sim

    input_file_list = list()
    dirs = label_path.split('/')[:-1]
    cur_path = ''
    for dir in dirs:
        cur_path = os.path.join(cur_path, dir)

    if not os.path.exists(os.path.join(cur_path, 'data')):
        os.makedirs(os.path.join(cur_path, 'data'))

    if not os.path.exists(os.path.join(cur_path, 'pairs')):
        os.makedirs(os.path.join(cur_path, 'pairs'))

    if os.path.isfile(label_path):
        input_file_list.append(label_path.split('/')[-1])
    else:
        if os.path.isdir(label_path):
            # add all txt files
            files = [f for f in os.listdir(label_path)
                     if os.path.isfile(os.path.join(label_path, f)) and f[-3:] == 'txt']
            input_file_list.extend(files)

    for label_file in input_file_list:
        #file_name = label_file.split('.')[0]
        #data_pairs_path = file_name + '_data.pkl'
        data_file_path = os.path.join(cur_path, 'data', label_file[:-3]+'pkl')
        input_file_path = os.path.join(cur_path, label_file)

        if not os.path.exists(data_file_path):
            with open(input_file_path) as fp:
                aids = []
                label_dict = {}
                for line in fp:
                    aid = int(line.strip().split("\t")[0])
                    label = int(line.strip().split("\t")[1])
                    aids.append(aid)
                    label_dict[aid] = label

                author_block = AuthorBlock(aids, False)

                # make a pandas data frame
                num_labels = len(aids)
                num_pairs = int(num_labels * (num_labels-1) * 0.5)

                data = pd.DataFrame(index=range(num_labels), columns=COLS)

                #for i in range(num_labels):
                #    data['aid'][i] = aids[i]

                # fill the data
                data['aid'] = aids

                for i in range(num_labels):
                    # fill informations
                    cur_author_point = author_block.get_points().get(aids[i])
                    cur_author = cur_author_point.get_author()
                    cur_doc = cur_author_point.get_doc()
                    cur_first_name = cur_author.get_first_name().lower()
                    cur_last_name = cur_author.get_last_name().lower()
                    cur_middle_name = cur_author.get_middle_name().lower()

                    data.loc[i, 'label'] = int(label_dict.get(aids[i]))

                    data.loc[i, 'fn_f'] = cur_first_name
                    data.loc[i, 'fn_1'] = cur_first_name[:1]
                    data.loc[i, 'fn_3'] = cur_first_name[:3]
                    data.loc[i, 'fn_5'] = cur_first_name[:5]
                    data.loc[i, 'fn_7'] = cur_first_name[:7]
                    data.loc[i, 'fn_b1'] = cur_first_name[-1:]
                    data.loc[i, 'fn_b3'] = cur_first_name[-3:]
                    data.loc[i, 'fn_b5'] = cur_first_name[-5:]
                    data.loc[i, 'fn_b7'] = cur_first_name[-5:]

                    data.loc[i, 'ln_f'] = cur_last_name
                    data.loc[i, 'ln_1'] = cur_last_name[:1]
                    data.loc[i, 'ln_3'] = cur_last_name[:3]
                    data.loc[i, 'ln_5'] = cur_last_name[:5]
                    data.loc[i, 'ln_7'] = cur_last_name[:7]
                    data.loc[i, 'ln_b1'] = cur_last_name[-1:]
                    data.loc[i, 'ln_b3'] = cur_last_name[-3:]
                    data.loc[i, 'ln_b5'] = cur_last_name[-5:]
                    data.loc[i, 'ln_b7'] = cur_last_name[-7:]

                    data.loc[i, 'mn_f'] = cur_middle_name
                    data.loc[i, 'mn_1'] = cur_middle_name[:1]
                    data.loc[i, 'mn_3'] = cur_middle_name[:3]
                    data.loc[i, 'mn_5'] = cur_middle_name[:5]
                    data.loc[i, 'mn_7'] = cur_middle_name[:7]
                    data.loc[i, 'mn_b1'] = cur_last_name[-1:]
                    data.loc[i, 'mn_b3'] = cur_last_name[-3:]
                    data.loc[i, 'mn_b5'] = cur_last_name[-5:]
                    data.loc[i, 'mn_b7'] = cur_last_name[-7:]

                    data.loc[i, 'affil'] = cur_author.get_affil().lower()

                    cur_order = cur_author.get_order()
                    if cur_order == 1:
                        data.loc[i, 'order'] = 1
                    elif cur_order == cur_doc.get_num_authors():
                        data.loc[i, 'order'] = 3
                    else:
                        data.loc[i, 'order'] = 2

                    cur_year = cur_doc.get_year()
                    data.loc[i, 'year_f'] = cur_year
                    data.loc[i, 'year_1'] = int(str(cur_year)[:1])
                    data.loc[i, 'year_2'] = int(str(cur_year)[:2])
                    data.loc[i, 'year_3'] = int(str(cur_year)[:3])
                    data.loc[i, 'journalid'] = cur_doc.get_journalid()
                    #data.loc[i, 'title_bow'] = cur_doc.title_bow

                data.info()
                print data[0:3]
                pd.to_pickle(data, data_file_path)

    data = None
    for label_file in input_file_list:
        data_file_path = os.path.join(cur_path, 'data', label_file[:-3]+'pkl')
        cur_data = pd.read_pickle(data_file_path)
        #cur_data.info()
        if data is not None:
            data = data.append(cur_data, ignore_index=True)
        else:
            data = cur_data

    data.info()
    data2 = data.copy()
    data_pairs = df_crossjoin(data, data2, suffixes=('_1', '_2'))
    data_pairs = pd.DataFrame(data_pairs.to_records())
    data_pairs = data_pairs[data_pairs['aid_1'] < data_pairs['aid_2']]
    data_pairs = data_pairs.reset_index(drop=True)

    data_pairs.info()
    print data_pairs[10:12][['aid_1', 'aid_2', 'fn_f_1', 'fn_f_2']]

    print 'cross join done'

    # data_pairs['match'] = False
    # for i in range(len(data_pairs.index)):
    #     if i% 100000 == 0:
    #         print i
    #     #print label_dict.get(data_pairs[i]['aid_1'])
    #     #if label_dict.get(data_pairs.loc[i, 'aid_1']) == label_dict.get(data_pairs.loc[i, 'aid_2']):
    #     #    data_pairs.loc[i, 'match'] = True
    #
    #     if data_pairs.loc[i, 'label_1'] == data_pairs.loc[i, 'label_2']:
    #         data_pairs.loc[i, 'match'] = True
    # print data_pairs[10:15][['aid_1', 'aid_2', 'label_1', 'label_2', 'fn_f_1', 'fn_f_2', 'match']]

    aids = []
    for label_file in input_file_list:
        input_file_path = os.path.join(cur_path, label_file)

        with open(input_file_path) as fp:
            for line in fp:
                aid = int(line.strip().split("\t")[0])
                aids.append(aid)

    author_block = AuthorBlock(aids, False)
    author_points = author_block.get_points()

    print 'load author blocks done'

    for rule in COLS_TO_USE:
        col_1 = "{}_1".format(rule)
        col_2 = "{}_2".format(rule)
        data_pairs[rule] = data_pairs[col_1] == data_pairs[col_2]

    print "simple comparisions done"

    data_pairs['match'] = data_pairs['label_1'] == data_pairs['label_2']

    data_pairs['fn_c'] = (data_pairs['fn_f_1'] == '') | \
                         (data_pairs['fn_f_2'] == '') | \
                         (
                             (
                                 (data_pairs['fn_1_1'] == data_pairs['fn_f_1']) |
                                 (data_pairs['fn_1_2'] == data_pairs['fn_f_2'])
                             )
                             &
                             (data_pairs['fn_1_1'] == data_pairs['fn_1_2'])
                         ) | \
                         (
                             (
                                (data_pairs['fn_1_1'] != data_pairs['fn_f_2']) &
                                (data_pairs['fn_1_2'] != data_pairs['fn_f_2'])
                             )
                             &
                             (data_pairs['fn_f_1'] == data_pairs['fn_f_2'])
                         )
    data_pairs['mn_c'] = (data_pairs['mn_f_1'] == '') | \
                         (data_pairs['mn_f_2'] == '') | \
                         (
                             (
                                 (data_pairs['mn_1_1'] == data_pairs['mn_f_1']) |
                                 (data_pairs['mn_1_2'] == data_pairs['mn_f_2'])
                             )
                             &
                             (data_pairs['mn_1_1'] == data_pairs['mn_1_2'])
                         ) | \
                         (
                             (
                                (data_pairs['mn_1_1'] != data_pairs['mn_f_2']) &
                                (data_pairs['mn_1_2'] != data_pairs['mn_f_2'])
                             )
                             &
                             (data_pairs['mn_f_1'] == data_pairs['mn_f_2'])
                         )
    data_pairs['ln_c'] = (data_pairs['ln_f_1'] == '') | \
                         (data_pairs['ln_f_2'] == '') | \
                         (
                             (
                                 (data_pairs['ln_1_1'] == data_pairs['ln_f_1']) |
                                 (data_pairs['ln_1_2'] == data_pairs['ln_f_2'])
                             )
                             &
                             (data_pairs['ln_1_1'] == data_pairs['ln_1_2'])
                         ) | \
                         (
                             (
                                (data_pairs['ln_1_1'] != data_pairs['ln_f_2']) &
                                (data_pairs['ln_1_2'] != data_pairs['ln_f_2'])
                             )
                             &
                             (data_pairs['ln_f_1'] == data_pairs['ln_f_2'])
                         )
    data_pairs['affil_c'] = (data_pairs['affil_1'] == '') | \
                            (data_pairs['affil_2'] == '') | \
                            (data_pairs['affil_1'] == data_pairs['affil_2'])
    # data_pairs['journalid_c'] = (data_pairs['journalid_1'] == '') | \
    #                             (data_pairs['journalid_2'] == '') | \
    #                             (data_pairs['journalid_1'] == data_pairs['journalid_2'])
    data_pairs['year_c'] = (data_pairs['year_f_1'] == 0) |\
                           (data_pairs['year_f_2'] == 0) |\
                           (data_pairs['year_f_1'] == data_pairs['year_f_2'])

    for diff in [2, 5, 10]:
        data_pairs["year_diff_{}".format(str(diff))] = abs(data_pairs['year_f_1'] - data_pairs['year_f_2']) < diff

    # data_pairs['title_bow'] = sum([data_pairs['title_bow_1'][k] *
    #                                data_pairs['title_bow_2'][k] for k in
    #                               data_pairs['title_bow_1'].viewkeys() &
    #                                data_pairs['title_bow_2'].viewkeys()]) > 0.5

    print "compatible done"

    title_dist = data_pairs.apply(lambda x: cos_dist(author_points.get(x['aid_1']).get_doc().title_bow,
                                                     author_points.get(x['aid_2']).get_doc().title_bow,
                                                     ), axis=1)
    print "title_dist done"

    affil_dist = data_pairs.apply(lambda x: cos_dist(author_points.get(x['aid_1']).get_author().affil_bow,
                                                     author_points.get(x['aid_2']).get_author().affil_bow,
                                                     ), axis=1)
    print "affil_dist done"

    venue_dist = data_pairs.apply(lambda x: cos_dist(author_points.get(x['aid_1']).get_doc().journal_bow,
                                                     author_points.get(x['aid_2']).get_doc().journal_bow,
                                                     ), axis=1)
    print "venue_dist done"

    coauth_dist = data_pairs.apply(lambda x: cos_dist(author_points.get(x['aid_1']).get_doc().author_bow,
                                                     author_points.get(x['aid_2']).get_doc().author_bow,
                                                     ), axis=1)
    print "coauth_dist done"

    for t in np.linspace(0.2, 0.8, 4):
        data_pairs["title_cos_{}".format(str(t))] = title_dist < t
        data_pairs["affil_cos_{}".format(str(t))] = affil_dist < t
        data_pairs["venue_cos_{}".format(str(t))] = venue_dist < t
        data_pairs["coauth_cos_{}".format(str(t))] = coauth_dist < t

        # data_pairs["title_cos_{}".format(str(t))] = data_pairs.apply(lambda x:
        #                                                              cos_dist(author_points.get(
        #                                                                  x['aid_1']).get_doc().title_bow,
        #                                                                       author_points.get(
        #                                                                           x['aid_2']).get_doc().title_bow,
        #                                                                       t), axis=1)
        # data_pairs["affil_cos_{}".format(str(t))] = data_pairs.apply(lambda x:
        #                                                              cos_dist(author_points.get(
        #                                                                  x['aid_1']).get_author().affil_bow,
        #                                                                       author_points.get(
        #                                                                           x['aid_2']).get_author().affil_bow,
        #                                                                       t), axis=1)
        # data_pairs["venue_cos_{}".format(str(t))] = data_pairs.apply(lambda x:
        #                                                              cos_dist(author_points.get(
        #                                                                  x['aid_1']).get_doc().journal_bow,
        #                                                                       author_points.get(
        #                                                                           x['aid_2']).get_doc().journal_bow,
        #                                                                       t), axis=1)
        # data_pairs["coauth_cos_{}".format(str(t))] = data_pairs.apply(lambda x:
        #                                                              cos_dist(author_points.get(
        #                                                                  x['aid_1']).get_doc().author_bow,
        #                                                                       author_points.get(
        #                                                                           x['aid_2']).get_doc().author_bow,
        #                                                                       t), axis=1)

    print "TF-IDF done"
    print data_pairs[10:15][['title_cos_0.4', 'match']]


    for rule in COLS_TO_USE:
        col_1 = "{}_1".format(rule)
        col_2 = "{}_2".format(rule)
        # we don't need the actual field values after this.
        data_pairs.drop([col_1, col_2], axis=1, inplace=True)

    pair_file_path = os.path.join(cur_path, 'pairs', 'pairs.pkl')
    pd.to_pickle(data_pairs, pair_file_path)

    # print len(data_pairs[data_pairs['match'] == True])


def df_crossjoin(df1, df2, **kwargs):
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1

    res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    res.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    return res

# def fill_pairwise_data(aids1, aids2):

if __name__ == '__main__':
    load_labels(sys.argv[1])
