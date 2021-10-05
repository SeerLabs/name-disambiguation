import numpy
from dao.author_block import AuthorBlock


class DBScan(object):
    """
    DBSCAN clustering algorithm
    """
    def __init__(self, block, eps, min_pts):
        self.block = block
        self.eps = eps
        self.min_pts = min_pts

        # author point id -> cluster id
        self.num_record = block.get_nauthors()

        # -1 means it is unvisited. This array will contain the final cluster index for each point
        self.cluster_id = numpy.array([-1] * self.num_record)

    def run(self):
        """ run DBSCAN clustering """
        '''
        def getNeighbors(idx):
            def comp(idx_1, idx_2):
                return int(self.mat_dist[idx_1, idx_2] - self.mat_dist[idx_1, idx_2])

            neighbors = []
            np.concatenate((np.where(data[4, 5:] > 0)[0], np.where(data[:4, 4] > 0)[0]))
            cand_idxes = numpy.where(self.mat_dist[idx] <= self.eps)[0]

            ncandis = cand_idxes.shape[0]
            if ncandis < self.min_pts:
                return neighbors
            else:
                # check constraint among each candidates(name / year)
                sorted_cand_idxes = sorted(cand_idxes, cmp=comp)
                neighbors.append(idx)

                for idx1 in sorted_cand_idxes:
                    if idx1 == idx:
                        continue
                    for neighbor in neighbors:
                        if self.mat_dist[idx1, neighbor] == 3.0:
                            break
                    else:
                        neighbors.append(idx1)
            return neighbors
        '''
        def apply_constraint(indices, members):
            filtered_indices = list()
            for idx in indices:
                for member in members:
                    # check name constraint only
                    #idx_check = AuthorBlock.get_idx_in_matrix(idx, member)
                    idx_check = (idx, member)
                    if self.block.get_distance_matrix()[idx_check] <= 0.0:
                        break
                else:
                    filtered_indices.append(idx)
            return filtered_indices

        def expand_cluster(p1, n1, cid):
            # find all current cluster members
            cluster_members = [p1]
            # idx = 0
            # for current_id in self.cluster_id:
            #     if current_id == cid:
            #         cluster_members.append(idx)
            #     idx += 1

            # list of ids to check
            seeds = n1
            selected_points = set(n1)

            i = 0
            while len(seeds) > 0:
                p2 = seeds.pop(0)
                # check if it is already in other cluster
                if self.cluster_id[p2] == 0:
                    self.cluster_id[p2] = cid
                    cluster_members.append(p2)
                # skip if the current one already assigned with different cluster
                elif self.cluster_id[p2] > 0:
                    continue
                else:
                    self.cluster_id[p2] = cid
                    cluster_members.append(p2)
                    n2 = self.block.get_neighbors(p2, self.eps, self.min_pts)
                    if n2 >= self.min_pts - 1:
                        n2_filtered = apply_constraint(n2, cluster_members)
                        if len(n2_filtered) >= self.min_pts - 1:
                            for p3 in n2_filtered:
                                #if self.cluster_id[p3] == 0:
                                #    self.cluster_id[p3] = cid
                                #    cluster_members.append(p3)
                                if p3 not in selected_points:
                                    seeds.append(p3)
                                    selected_points.add(p3)

                # if self.cluster_id[p2] == -1:
                #     n2 = getNeighbors(p2)
                #     if len(n2) >= self.min_pts:
                #         # remove itself
                #         n2 = n2[1:]
                #         # Calculate cluster constraint
                #         n2_filtered = apply_constraint(n2, cluster_members)
                #         # check if we have more than min_pts-1 neighbors (excluding itself)
                #         if len(n2_filtered) >= self.min_pts - 1:
                #             for p3 in n2_filtered:
                #                 if self.cluster_id[p3] == -1 and p3 not in selected_points:
                #                     seeds.append(p3)
                #                     selected_points.add(p3)
                # if self.cluster_id[p2] <= 0:
                #     self.cluster_id[p2] = cid
                #     cluster_members.append(p2)

                i += 1
                #seeds = seeds[1:]

        # start of the code..
        cur_id = 0
        # cnt = 0
        # for each point p in the records
        for p1 in range(self.num_record):
            # cnt += 1
            # if cnt % 100 == 0:
            #     print str(cnt) + " / " + str(self.num_record) + " done"

            # check if it is previously processed
            if self.cluster_id[p1] >= 0:
                continue
            else:
                # find all neighbors
                n1 = self.block.get_neighbors(p1, self.eps, self.min_pts)
                # print p1, n1
                # density check
                if len(n1) < self.min_pts-1:
                    # label as noise
                    self.cluster_id[p1] = 0
                else:
                    cur_id += 1
                    # label itself with the new cluster id
                    self.cluster_id[p1] = cur_id
                    expand_cluster(p1, n1, cur_id)
                    # print numpy.where(self.cluster_id == cur_id)

    def write_result(self, outpath):
        cid = 1
        #arr = numpy.asarray(self.cluster_id)
        arr = self.cluster_id
        points = self.block.get_all_aids()
        with open(outpath, 'w+') as ofp:
            while True:
                idxes = numpy.where(arr == cid)[0]
                if idxes.shape[0] == 0:
                    break
                ofp.write('[' + str(cid) + ']\n')
                for idx in idxes:
                    ofp.write(str(points[idx]) + '\n')
                cid +=1

            idxes = numpy.where(arr == 0)[0]
            if idxes.shape[0] > 0:
                ofp.write('[noise]\n')
                for idx in idxes:
                    ofp.write(str(points[idx]) + '\n')

