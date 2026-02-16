import numpy as np
import time
import pathlib

from utils import read, index, calc, plot

# Index search based on KD-tree
# Mentioned in https://www.researchgate.net/publication/221597836_Indexing_Biometric_Databases_Using_Pyramid_Technique

def run(
        pca=True,
        seed=55,
        show_plots=False,
):
    run_params = locals()
    np.random.seed(seed)

    # Load data
    gallery, probe, _, d_table = read.load()

    if pca:
        normal = True
        n_comp = 100  # 260
        gallery.samples, expl_gallery = calc.pca(gallery.samples, normal=normal, n_comp=n_comp)
        probe.samples, expl_probe = calc.pca(probe.samples, normal=normal, n_comp=n_comp)
        if show_plots:
            plot.explained_variance(expl_gallery, "gallery")
            plot.explained_variance(expl_probe, "probes")

    # Build index structure from gallery samples
    kd_tree = index.KDTree()
    build_time = time.time()
    kd_tree.build(gallery.samples, labels=np.arange(gallery.n))
    build_time = time.time() - build_time
    print(f"time to build the index structure: {build_time:.2f} s")

    # Identify all probe samples
    y_true = np.array(d_table.loc[:, "gallery_idx"])  # true gallery index for each probe sample
    identify_time = time.time()
    y_pred_list = kd_tree.identify(probe.samples)
    y_bin = [(y_true[i] in y_pred_list[i]) for i in range(probe.n)]
    identify_time = time.time() - identify_time
    print(f"time to identify all probe samples: {identify_time:.2f} s")
    # get number of retrieved gallery identites for each probe querry
    y_ret = [len(y_pred_list[i]) for i in range(probe.n)]

    # Evaluation
    hit_rate = np.mean(y_bin)
    print(f"=> hit rate: {hit_rate:.4f}")
    pen_rate = np.mean(y_ret) / gallery.n
    print(f"=> penetration rate: {pen_rate:.4f}")
    if show_plots: plot.retrieved_number(y_ret)

    results = {
        "approach": "Indexing with KD-tree",
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
