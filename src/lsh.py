import numpy as np
import time
import pathlib

from utils import read, index, calc, plot


# Locality Sensitive Hashing (LSH) with Random Projections
# Inspired by: https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23

def run(
        num_tables=5,
        hash_size=2,
        pca=False,
        seed=55,
        show_plots=True,
):
    run_params = locals()
    np.random.seed(seed)

    # Load data
    gallery, probe, _, d_table = read.load()
    if pca:
        normal = True
        n_comp = 260
        gallery.samples, expl_gallery = calc.pca(gallery.samples, normal=normal, n_comp=n_comp)
        probe.samples, expl_probe = calc.pca(probe.samples, normal=normal, n_comp=n_comp)
        if show_plots:
            plot.explained_variance(expl_gallery, "gallery")
            plot.explained_variance(expl_probe, "probes")
    
    # Build index structure from gallery samples
    print(f"num_tables: {num_tables}")
    print(f"hash_size: {hash_size}")
    lsh = index.LSH(num_tables=num_tables, hash_size=hash_size, inp_dimensions=gallery.samples.shape[-1])
    build_time = time.time()
    lsh.build(gallery.samples, labels=np.arange(gallery.n))
    build_time = time.time() - build_time
    print(f"time to build the index structure: {build_time:.2f} s")

    # Identify all probe samples
    y_true = np.array(d_table.loc[:, "gallery_idx"])  # true gallery index for each probe sample
    identify_time = time.time()
    # TODO: this approach cannot know whether the correct identity is even contained in the returned prediction (-> performance is even worse)
    y_pred_list = lsh.identify(probe.samples)
    y_bin = [(y_true[i] in y_pred_list[i]) for i in range(probe.n)]
    identify_time = time.time() - identify_time
    # get number of retrieved gallery identites for each probe querry
    y_ret = [len(y_pred_list[i]) for i in range(probe.n)]
    print(f"time to identify all probe samples: {identify_time:.2f} s")

    # Evaluation
    hit_rate = np.mean(y_bin)
    print(f"=> hit rate: {hit_rate:.4f}")
    pen_rate = np.mean(y_ret) / gallery.n
    print(f"=> penetration rate: {pen_rate:.4f}")
    if show_plots: plot.retrieved_number(y_ret)

    results = {
        "approach": "Random Projection LSH",
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
