import numpy as np
import sklearn
from bitarray.util import count_xor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cos_sim_vec(x, y):
    # Cosine similarity of two vectors
    cs = (x @ y.T) / (np.linalg.norm(x) * np.linalg.norm(y))
    return cs


def cos_sim(X, Y):
    # Cosine similarity of two matrices
	return cosine_similarity(X, Y, dense_output=True)


def disimilarity(current, reference):
	# Calculate the disimilarity between two bitarrays
	# according to https://ieeexplore.ieee.org/document/7139105

	numer = count_xor(current, reference)
	denom = current.count(1) + reference.count(1)
	return numer / denom


def hit_rate(y_true, y_pred):
    # Calculate the hit rate (corresponds to the accuracy/rank 1 recognition)
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def youden_index(sensitivity, specificity):
    # Youden's index is maximized to find the best trade-off between sensitivity and specificity
	return (sensitivity + specificity - 1)


def euklidean_dist(a, b):
	# Euklidean distance between two n-dimensional points a and b
	return np.linalg.norm(np.array(a)-np.array(b))


def filter_params(run_params, blacklist=["seed", "show_plots"]):
    # Remove blacklisted key-value pairs from the run_params dictionary (shallow)

    for elem in blacklist:
        if elem in run_params:
            run_params.pop(elem)
    return run_params


def normalize(x, method):
	if method == "stdscaler":
		# Normalize to zero mean and unit variance
		std = StandardScaler()
		return std.fit_transform(x)
	elif method == "minmax":
		# Normalize to range 0 to 1 for each feature
		minmax = MinMaxScaler()
		return minmax.fit_transform(x)
	else:
		raise ValueError(f"Method {method} for normalization not supported")


def pca(x, normal, n_comp=None):
	# Apply Principal Component Analysis (PCA)
	if normal:
		x = normalize(x, method="stdscaler")
	if n_comp is None:
		n_comp = x.shape[-1]
	print("n_comp:", n_comp)
	pca = PCA(n_components=n_comp)
	x_pca = pca.fit_transform(x)
	expl = pca.explained_variance_ratio_
	return x_pca, expl
