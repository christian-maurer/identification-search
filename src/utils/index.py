import numpy as np
import math
from sklearn.cluster import KMeans
from bitarray import bitarray

from . import calc


# Source: https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23
class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)  # TODO: add seed?
        
    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, []) + [label]
        
    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])  # returns emtpty list if hash_value is not found


# Source: https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23
class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))
    
    def __setitem__(self, inp_vec, label):
        # Add an identity label to LSH table at a place according to the feature vector: LSH[inp_vec] = label
        for table in self.hash_tables:
            table[inp_vec] = label
    
    def __getitem__(self, inp_vec):
        # Retrieve labels of identies with similar feature vector: LSH[inp_vec]
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])

        """ no improvement:
        t = 2
        labels, counts = np.unique(results, return_counts=True)
        t_times_labels = [label for label, count in zip(labels, counts) if count >= t]
        return t_times_labels
        """

        return list(set(results))
    
    def build(self, samples, labels):
        # Build the LSH Tables with samples and corresponding labels
        for s, l in zip(samples, labels):
            self.__setitem__(s, l)  # self[s] = l
    
    def identify(self, probes):
        # For each probe in samples get a list of labels whose feature vectors are similar
        results = list()
        for p in probes:
            r = self.__getitem__(p)
            results.append(r)
        return results


class Clustering:
    def __init__(self, n_clusters, seed):
        self.hash_table = dict()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init="auto",
            max_iter=300,
            tol=1e-4,
            verbose=0,
            random_state=seed,
            copy_x=True,
            algorithm="lloyd",
        )

    def build(self, gallery, labels):
        # Fit the k-Means estimator with gallery samples
        # and build the hash table with the predicted centers and corresponding labels of the gallery samples
        # Note that multiple gallery samples can have the same center if there are less clusters than gallery samples

        self.kmeans.fit(gallery)
        gallery_centers = self.kmeans.predict(gallery)
        for center, label in zip(gallery_centers, labels):
            if not center in self.hash_table:
                self.hash_table[center] = [label]
            else:
                self.hash_table[center].append(label)

    def identify(self, probes, all_at_once=False):
        # Get a prediction for each probe to which cluster-center it is assigned to
        # and return the labels of the identities that are in the same cluster

        if not all_at_once:
            results = list()
            for p in probes:
                center = self.kmeans.predict([p])[0]  # predict for each probe individually in sequence for a more realistic scenario
                results.append(self.hash_table[center])
            return results
        else:
            probe_centers = self.kmeans.predict(probes)
            results = list()
            for center in probe_centers:
                results.append(self.hash_table[center])
            return results


class BloomTree:
    def __init__(self, resolution):
        self.tree = list()  # binary tree of bloom filters where tree[0] is the root, tree[1] the left child, tree[2] the right child, etc.
        self.labels = dict()
        self.resolution = resolution
    
    def build(self, gallery, labels):
        gallery_norm = calc.normalize(gallery, method="minmax")
        current_samples_list = list()
        current_samples_list.append(gallery_norm)
        nodes_on_lvl = [2**i for i in range(math.ceil(np.log2(len(gallery_norm))) + 1)]
        #print("elements of nodes on progressing lvl", nodes_on_lvl)
        for lvl in range(len(nodes_on_lvl)):
            begin_level_idx = (2 ** lvl) - 1
            next_samples_list = list()
            counter = 0
            for i, samples in enumerate(current_samples_list):
                bf = self.bloom_filter(samples)  # samples is a list of samples
                self.tree.append(bf)
                if len(samples) == 1 and lvl == len(nodes_on_lvl) - 1:  # if node has one sample and is a leaf (on last level)
                    node_idx = begin_level_idx + i                    
                    self.labels[node_idx] = labels[counter]  # maps leaf node_idx to a label (gallery_idx)
                    counter += 1
                next_samples_list.extend(np.array_split(samples, 2))
            current_samples_list = next_samples_list
    
    def identify(self, probes):
        probes_norm = calc.normalize(probes, method="minmax")
        results = list()
        for p in probes_norm:
            idx = self.leaf_idx(self.bloom_filter([p]))
            results.append(idx)
        return results
    
    def leaf_idx(self, bf):
        # Climb down the tree and find the leaf with the lowest disimilarity

        node = 0
        while True:
            left = 2 * node + 1
            right = 2 * node + 2
            if left >= len(self.tree) or right >= len(self.tree) or self.tree[left] is None:
                return self.labels[node]
            if self.tree[right] is None:
                node = left
                continue
            disim_left = calc.disimilarity(bf, self.tree[left])
            disim_right = calc.disimilarity(bf, self.tree[right])
            node = left if disim_left < disim_right else right

    def bloom_filter(self, samples):
        # Return a bloom filter (bitarray) from one or more samples

        if len(samples) == 0:
            return None
        idxs = np.interp(samples, [0, 1], [0, self.resolution - 1]).astype(int)  # for each sample assign each feature a value between [0, resolution - 1]
        bf = np.zeros((samples[0].size, self.resolution), dtype=bool)
        for idx in idxs:
            temp = np.zeros((idx.size, self.resolution), dtype=bool)
            temp[np.arange(idx.size), idx] = 1  # one hot encoding for each feature value [0, resolution - 1]
            bf |= temp  # logical OR to combine bloom filters from all sampels
        x = bf.flatten().astype(int).tolist()  # combine feature and resolution dimension to x.shape=(n_samples * resolution)
        bf = bitarray(x)
        return bf

    '''
    @staticmethod
    def bloom_filter_old(samples):
        # Note: this is not an actual bloom filter!

        if len(samples) == 0:
            return None
        
        # Initialize bloomfilter with size n and zeros
        bf = bitarray(samples[0].shape[-1])
        bf.setall(False)

        for s in samples:
            temp = np.array((s >= 0.5)).astype(int).tolist()
            bf |= bitarray(temp)
        return bf
    '''


class KDTree:
    def __init__(self):
        self.thresholds = list()  # binary tree containing thresholds for each node where thresholds[0] is the root, thresholds[1] the left child, thresholds[2] the right child, etc.
        self.labels = list()

    def build(self, gallery, labels):
        current_samples_list = list()
        current_samples_list.append(gallery)
        for lvl in range(gallery.shape[1]):

            #begin_level_idx = (2 ** lvl) - 1
            next_samples_list = list()
            for i, samples in enumerate(current_samples_list):
                #node_idx = begin_level_idx + i
                median = None
                left_idx = []
                right_idx = []
                if not len(samples) == 0:  # if node has no samples

                    median = np.median(samples[:, lvl])
                    sel_feature = np.array(samples[:, lvl])
                    left_idx = np.where(sel_feature <= median)[0]
                    right_idx = np.where(sel_feature > median)[0]
                    assert len(left_idx) + len(right_idx) == len(samples)

                    left_samples = samples[left_idx] if not len(left_idx) == 0 else []
                    right_samples = samples[right_idx] if not len(right_idx) == 0 else []
                    next_samples_list.extend([left_samples, right_samples])
                self.thresholds.append(median)
                self.labels.extend([left_idx, right_idx])  # TODO: use labels param
            current_samples_list = next_samples_list

    def identify(self, probes):
        results = list()
        for p in probes:
            idx = self.leaf_idx(p)
            results.append(idx)
        return results

    def leaf_idx(self, p):
        # Climb down the tree and find the leaf
        node = 0
        lvl = 0
        while True:
            left = 2 * node + 1
            right = 2 * node + 2
            if left >= len(self.thresholds) or right >= len(self.thresholds) \
                    or self.thresholds[left] is None or self.thresholds[right] is None:
                return self.labels[node]

            node = left if p[lvl] < self.thresholds[node] else right
            lvl += 1
