from glob import glob
import random
import sys
import numpy as np
from tqdm import tqdm


def run(input_path):
    file_list = list()
    for filename in glob(input_path + '/*.out'):
        file_list.append(filename)

    # load labels
    with open(file_list[0], 'r') as fp:
        for line in fp:
            data_label = line.strip().split(',')
            break

    nfeat = len(data_label)

    def get_feature_matrix(file_list):
        data_list = list()
        nrows = 0
        for file_name in tqdm(file_list):
            cur_dataset = np.genfromtxt(file_name, delimiter=',',
                                        dtype='float', skip_header=1)
            # if cur_dataset.ndim == 1:
            #     cur_dataset = np.reshape(cur_dataset, (1, cur_dataset.shape[0]))

            # print dataset_train.shape
            comp = np.load(file_name[:-4] + '_comp.npy')
            comp = comp.astype(np.float32)
            comp = np.reshape(comp, (comp.shape[0], 1))

            cur_dataset = np.hstack((cur_dataset, comp))

            nrows += cur_dataset.shape[0]
            data_list.append(cur_dataset)

        dataset = np.zeros((nrows, nfeat+1), dtype=np.float32)
        #dataset = np.zeros((nrows, nfeat), dtype=np.float32)

        cur_idx = 0
        for data in data_list:
            dataset[cur_idx:cur_idx+data.shape[0]] = data
            cur_idx += data.shape[0]
        return dataset

    dataset = get_feature_matrix(file_list)
    print "num positive : " + str(len(np.where(dataset[:, -1] == 1.0)[0]))
    print "num negative : " + str(len(np.where(dataset[:, -1] == 0.0)[0]))
    print "num total : " + str(dataset.shape[0])


if __name__ == "__main__":
    trainPath = sys.argv[1]
    run(trainPath)
