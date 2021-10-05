"""
Extract features from labeled data file(s)
"""
import os
import sys
import timeit
from multiprocessing import Pool
import numpy

from dao.author_block import AuthorBlock

NTHREADS = 20

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

    with open(outpath, 'w+') as ofp:
        # ofp.write("auth_fst,auth_mid,auth_ord," +
        #           "year,year_diff,mesh_shared,mesh_shared_idf,title_shared," +
        #           "aff_jac,aff_jw,coauth_shared,coauth_idf,coauth_jac," +
        #           "jour_shared_idf,label\n")
        ofp.write("title_idf,affil_idf,grant_idf,journal_idf,chemical_idf,mesh_idf," +
                  # "author_idf,year_diff,abstract_idf,label\n")
                  "author_idf,year_diff,abstract_idf,title_w2v,abstract_w2v,label\n")
        compatible_array = block.calc_features_and_write(ofp, gold_dic)
        comp_path = outpath[:-4] + '_comp.npy'
        numpy.save(comp_path, compatible_array)


def run(indir, outdir):
    for dirpath, dirs, files in os.walk(indir):
        file_pathes = list()
        out_pathes = list()

        for file in files:
            file_path = os.path.join(dirpath, file)
            out_path = os.path.join(outdir, file.replace(".txt", ".out"))
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


# def run(infile, indir, outdir):
#     file_pathes = list()
#     out_pathes = list()
#
#     with open(infile, 'r') as fp:
#         for line in fp:
#             file_name = line.strip()
#             cur_path = os.path.join(indir, file_name)
#             out_path = os.path.join(outdir, file_name.replace('.txt', '.out'))
#             file_pathes.append(cur_path)
#             out_pathes.append(out_path)
#
#     args = []
#     for i in range(len(file_pathes)):
#         args.append((file_pathes[i], out_pathes[i]))
#
#     start = timeit.default_timer()
#
#     pool = Pool(NTHREADS)
#     pool.map(helper_func, args)
#
#     #for i in range(len(file_pathes)):
#     #    write_features(*args[i])
#
#     end = timeit.default_timer()
#
#     print 'processing time : ' + str(end - start) + ' s'


if __name__ == '__main__':
    arg_count = len(sys.argv) -1
    if arg_count == 2:
        run(sys.argv[1], sys.argv[2])
    if arg_count == 3:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
