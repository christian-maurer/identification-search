import numpy as np
import time
import pathlib

from utils import read, index, plot


# Indexing with clustering based on the k-means machine learning model
# Inspired by: https://www.researchgate.net/publication/228364027_Efficient_search_and_retrieval_in_biometric_databases
# and https://randorithms.com/2019/09/19/Visual-LSH.html

def run(
        n_clusters=3531,
        seed=55,
        show_plots=True,
):
    run_params = locals()
    np.random.seed(seed)

    # Load data
    gallery, probe, _, d_table = read.load()
    
    # Build index structure from gallery samples
    print(f"n_clusters: {n_clusters}")
    clustering = index.Clustering(n_clusters=n_clusters, seed=seed)
    build_time = time.time()
    clustering.build(gallery.samples, labels=np.arange(gallery.n))
    build_time = time.time() - build_time
    print(f"time to build the index structure: {build_time:.2f} s")

    # Identify all probe samples
    y_true = np.array(d_table.loc[:,"gallery_idx"])  # true gallery index for each probe sample
    identify_time = time.time()
    y_pred_list = clustering.identify(probe.samples, all_at_once=False)
    y_bin = [y_true[i] in y_pred_list[i] for i in range(probe.n)]
    identify_time = time.time() - identify_time
    # Get number of retrieved gallery identites for each probe sample
    y_ret = [len(y_pred_list[i]) for i in range(probe.n)]
    print(f"time to identify all probe samples: {identify_time:.2f} s")

    # Evaluation
    hit_rate = np.mean(y_bin)
    print(f"=> hit rate: {hit_rate:.4f}")
    pen_rate = np.mean(y_ret) / gallery.n
    print(f"=> penetration rate: {pen_rate:.4f}")
    if show_plots: plot.retrieved_number(y_ret)

    results = {
        "approach": "Clustering with K-Means",
        "script": pathlib.Path(__file__).name,
        "run_params": run_params,
        "results": {
            "hit_rate": hit_rate,
            "pen_rate": pen_rate,
            "build_time": build_time,
            "identify_time": identify_time,
        }
    }
    return results


if __name__ == '__main__':
    run()
