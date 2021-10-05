"""
Extract features from labeled data file(s)
"""
import os
import sys
import timeit
from multiprocessing import Pool

from dao.author_block import AuthorBlock

NTHREADS = 16


def helper_func(args):
    return write_features(*args)


def write_features(inpath, outpath):
    gold_dic = {}
    gold_aids = []

    with open(inpath) as fp:
        for line in fp:
            data = line.split('\t')
            gold_dic[int(data[0])] = int(data[1])
            gold_aids.append(int(data[0]))

    print "Calculating Features of " + inpath + "..."
    block = AuthorBlock(gold_aids, False)
    block.extract_metadata(outpath, gold_dic)


def run(indir, outdir):
    for dirpath, dirs, files in os.walk(indir):
        file_pathes = list()
        out_pathes = list()

        for file in files:
            file_path = os.path.join(dirpath, file)
            out_path = os.path.join(outdir, file.replace(".txt", ".pkl"))
            file_pathes.append(file_path)
            out_pathes.append(out_path)

    args = []
    for i in range(len(file_pathes)):
        args.append((file_pathes[i], out_pathes[i]))

    start = timeit.default_timer()

    pool = Pool(NTHREADS)
    pool.map(helper_func, args)

    #for i in range(len(file_pathes)):
    #    write_features(*args[i])

    end = timeit.default_timer()

    print 'processing time : ' + str(end-start)  + ' s'

if __name__ == '__main__':
    arg_count = len(sys.argv) -1
    if arg_count == 2:
        run(sys.argv[1], sys.argv[2])
