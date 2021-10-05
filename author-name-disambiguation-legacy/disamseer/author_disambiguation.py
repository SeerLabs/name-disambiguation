import glob
import os
import sys
from datetime import datetime

import numpy as np
from dao.author_block import AuthorBlock

from clustering.dbscan import DBScan


class AuthorDisambiguation(object):
    """
    Class to run author name disambiguation
    """
    EPS = 0.85
    MINPTS = 2

    def __init__(self):
        self.points = {}
        self.block = None

    def disambiguate_file(self, cur_path, out_path, matrix_path=None):
        file_name = cur_path.split('/')[-1].strip()

        # make output path if not exists
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print 'START ' + file_name + ': ' + str(datetime.now())
        outpath = os.path.join(out_path, file_name.replace(".txt", ".out"))

        #print file_name + ': load block... ' + str(datetime.now())
        block = AuthorBlock(cur_path, True)

        #print file_name + ': calc distance matrix... ' + str(datetime.now())
        block.calc_distance_matrix()

        minpts = AuthorDisambiguation.MINPTS
        if len(block.aids) <= AuthorDisambiguation.MINPTS:
            minpts = 2

        #print file_name + ': run DBSCAN... ' + str(datetime.now())
        # do clustering with our own implementation
        dbscan = DBScan(block, AuthorDisambiguation.EPS, minpts)
        dbscan.run()
        dbscan.write_result(outpath)

        '''
        # do clustering
        # clustering = DBSCAN(eps=AuthorDisambiguation.EPS,min_samples=AuthorDisambiguation.MINPTS,metric='precomputed')
        # clustering = hdbscan.HDBSCAN(min_cluster_size=3, metric=calcDist)
        aids = block.aids
        #self.points = block.points
        y = clustering.fit_predict(mat_dist)
        label_dict = {}
        idx = 0
        for label in y:
            if not label_dict.get(label):
                list = []
                label_dict[label] = list
            else:
                list = label_dict.get(label)
            list.append(aids[idx])
            idx+=1
        #print y

        with open(outpath, 'w+') as ofp:
            idx = 1
            for label in label_dict:
                list = label_dict.get(label)
                if label != -1:
                    ofp.write('['+str(idx)+']\n')
                    idx += 1
                else:
                    ofp.write('[noise]\n')
                for aid in list:
                    ofp.write(str(aid)+'\n')
        '''

        print 'END ' + file_name + ': ' + str(datetime.now())

        if len(matrix_path) > 0:
            # write matrix
            matrix_file_path = os.path.join(matrix_path, file_name)
            np.save(matrix_file_path, block.get_distance_matrix())

    def run_disambiguation(self, in_path, out_path, matrix_path=''):
        """
        run disambiguation
        :param in_path: input file path
        :param out_path: output path
        :param matrix_path: dist matrix path(optional)
        """
        if not os.path.isfile(in_path):
            for filename in glob.glob(in_path + '/*.txt'):
                #cur_path = os.path.join(in_path, filename)
                self.disambiguate_file(filename, out_path, matrix_path)
        else:
            self.disambiguate_file(in_path, out_path, matrix_path)


if __name__ == '__main__':
    disambiguation = AuthorDisambiguation()
    disambiguation.run_disambiguation(sys.argv[1], sys.argv[2])


