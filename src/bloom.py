import numpy as np
import time
import pathlib

from utils import read, index, calc, plot


# Indexing with Bloom Filter Tree Structure
# Inspired by: https://ieeexplore.ieee.org/document/7139105

def run(
        pca=False,
        seed=55,
        show_plots=True,
):
    resolution=100,
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
    
    # Build Bloom filter tree structure from gallery samples
    bloom = index.BloomTree(resolution=resolution)
    build_time = time.time()
    bloom.build(gallery.samples, labels=np.arange(gallery.n))
    build_time = time.time() - build_time
    print(f"time to build the index structure: {build_time:.2f} s")

    # Make prediction on probes
    y_true = np.array(d_table.loc[:, "gallery_idx"])  # true gallery index for each probe sample
    identify_time = time.time()
    y_pred = bloom.identify(probe.samples)
    identify_time = time.time() - identify_time
    print(f"time to identify all probe samples: {identify_time:.2f} s")

    # Evaluation
    hit_rate = calc.hit_rate(y_true=y_true, y_pred=y_pred)
    print(f"=> hit rate: {hit_rate:.4f}")
    pen_rate = 1 / gallery.n
    print(f"=> penetration rate: {pen_rate:.4f}")

    results = {
        "approach": "Indexing with Bloom Filters",
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
