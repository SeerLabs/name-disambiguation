"""
Runs author disambiguation in parallel
"""
import os
import sys
from datetime import datetime
from multiprocessing import Pool

from author_disambiguation import AuthorDisambiguation


def helper_func(input_args):
    # print 'run disambiguation for ' + input_args[0] + '...'
    return run_disambiguation(*input_args)


def run_disambiguation(in_path, out_path, matrix_path=''):
    disamb = AuthorDisambiguation()
    disamb.run_disambiguation(in_path, out_path, matrix_path)


def run(indir, outdir, nthreads, matrix_dir=''):
        filepathes = list()
        outpathes = list()
        if len(matrix_dir):
            matrixpathes = list()

        for dirpath, dirs, files in os.walk(indir):
            for f in files:
                filepath = os.path.join(dirpath, f)
                outpath = outdir
                filepathes.append(filepath)
                outpathes.append(outpath)
                if len(matrix_dir):
                    matrixpathes.append(matrix_dir)

        args = []
        i = 0
        while i < len(filepathes):
            if len(matrix_dir):
                args.append((filepathes[i], outpathes[i], matrixpathes[i]))
            else:
                args.append((filepathes[i], outpathes[i]))
            i += 1

        pool = Pool(nthreads)
        pool.map(helper_func, args)

def run_from_file(infile, outdir, nthreads, matrix_dir=''):
    filepathes = list()
    outpathes = list()
    if len(matrix_dir):
        matrixpathes = list()

    with open(infile) as fp:
        for line in fp:
            file_name = line.split('\t')[0].strip()
            filepathes.append(file_name)

            #dir_name = line.split('/')[1].strip()
            #out_dir = os.path.join(outdir, dir_name)
            #outpathes.append(out_dir)
            outpathes.append(outdir)

            if len(matrix_dir):
                matrixpathes.append(matrix_dir)

        args = []
        i = 0
        while i < len(filepathes):
            if len(matrix_dir):
                args.append((filepathes[i], outpathes[i], matrixpathes[i]))
            else:
                args.append((filepathes[i], outpathes[i]))
            i += 1

        pool = Pool(nthreads)
        pool.map(helper_func, args)


if __name__ == '__main__':
    print 'START ' + str(datetime.now())
    #run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    run_from_file(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    print 'END ' + str(datetime.now())
