import numpy as np
import time
import pathlib
from tqdm import trange

from utils import read, calc, plot


# Baseline algorithm with no index structure

def min_dist(gallery, probes, all_at_once=False):
    # Calculate the cosine distance between each probe sample and all gallery samples
    # Return the index of the gallery sample with the minimun distance for each probe

    n = probes.shape[0]
    if not all_at_once:
        # Sequential implementation that calculates for each probe sample the gallery index that has a minimal distance individually
        min_dist_idx = np.zeros((n))
        print("Calculating cosine distance for each probe sample...")
        for i in trange(probes.shape[0]):
            #if (i % int(n / 30)) == 0:
            #    print(f"\rProgress {i / n * 100:.2f} %\r", end="")
            dist = 1 - calc.cos_sim(gallery, np.atleast_2d(probes[i]))[:, 0]
            min_dist_idx[i] = np.argmin(dist)
        return min_dist_idx
    else:
        # Fast implementation that calculates the similarity for between all probe and all gallery samples at once
        dist = 1 - calc.cos_sim(gallery, probes)
        min_dist_idx = np.argmin(dist, axis=0)  # extract indexes with the lowest distance
        return min_dist_idx


def main(
        random_indexing=False,
        random_indexing_iterations=6,
        seed=55,
        show_plots=True,
):
    run_params = locals()
    np.random.seed(seed)

    # Load data
    gallery, probe, _, d_table = read.load()
    
    if not random_indexing:
        # Baseline with Exhaustive Search

        # Identify all probe samples with the index structure
        identify_time = time.time()
        y_pred = min_dist(gallery.samples, probe.samples, all_at_once=False)
        identify_time = time.time() - identify_time
        print(f"time to identify all probe samples: {identify_time:.2f} s")
        y_true = np.array(d_table.loc[:, "gallery_idx"])  # true gallery index for each probe sample
        
        # Evaluation
        hit_rate = calc.hit_rate(y_true=y_true, y_pred=y_pred)
        print(f"=> hit rate: {hit_rate:.4f}")
        pen_rate = 1
        print(f"=> penetration rate: {pen_rate:.4f}")

        results = {
            "approach": "Baseline with Exhaustive Search",
            "script": pathlib.Path(__file__).name,
            "run_params": run_params,
            "results": {
                "hit_rate": hit_rate,
                "pen_rate": pen_rate,
                "build_time": 0,  # or np.nan?
                "identify_time": identify_time,
            }
        }
        return results

    else:
        # Baseline with Random Indexing
        gallery_sizes = np.array(np.linspace(1, gallery.n, random_indexing_iterations), dtype=int)
        hit_rates = []
        pen_rates = []
        identify_times = []
        for size in gallery_sizes:
            print(f"\nrunning with a random gallery subset of {size} samples:")

            # Choose a random subset of gallery samples
            gallery_indexes = np.arange(0, gallery.n, step=1)
            gallery_subset_idx = np.random.choice(gallery_indexes, size=size, replace=False)
            gallery_subset = gallery.samples[gallery_subset_idx]
            
            # Identification of all probe samples based a subset of gallery samples
            identify_time = time.time()
            y_pred_subset = min_dist(gallery_subset, probe.samples, all_at_once=True)  # returns labels that correspond with the labels for the chosen gallery subset
            identify_time = time.time() - identify_time
            identify_times.append(identify_time)
            # Take the identification time with caution if the fast implementation of min_dist function with all_at_once=True is used
            print(f"time to identify all probe samples (all probes identified at once!)): {identify_time:.2f} s")
            y_true = np.array(d_table.loc[:, "gallery_idx"])  # true gallery index for each probe sample considering all gallery sampels
            y_pred = [gallery_subset_idx[p] for p in y_pred_subset]  # re-label the predictions so they correspond with labels of the whole gallery (for compatibility with y_true)

            # Evaluation
            hit_rate = calc.hit_rate(y_true=y_true, y_pred=y_pred)
            hit_rates.append(hit_rate)
            print(f"=> hit rate: {hit_rate:.4f}")
            pen_rate = size / gallery.n
            pen_rates.append(pen_rate)
            print(f"=> penetration rate: {pen_rate:.4f}")

        if show_plots: plot.hit_vs_pen(hit_rates, pen_rates)
        
        results = {
            "approach": "Baseline with Random Indexing",
            "script": pathlib.Path(__file__).name,
            "run_params": run_params,
            "results": {
                "gallery_sizes": gallery_sizes,
                "hit_rate": hit_rates,
                "pen_rate": pen_rates,
                "build_time": [0] * len(gallery_sizes),  # or [np.nan]...?
                "identify_time": identify_times,
            }
        }
        return results


if __name__ == '__main__':
    main()
