import numpy as np
import pathlib
import sys
import os
import time
import json
import datetime
import joblib
import copy

from tqdm import tqdm

import lsh  # required although marked as not used
import clustering  # required although marked as not used
from utils import calc, plot


MEM = joblib.Memory(location="cache", verbose=0)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        #return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# Source: https://github.com/seemoo-lab/myo-keylogging
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return " ".join(str(obj).replace("\n", "").split())


def search(target, param_space):
    # Collect results of the runs with configurations from the parameter space

    results = {
          "approach": "Parameter search",
          "script": pathlib.Path(__file__).name,
          "target": target,
          "iterations": []
    }
    search_time = time.time()
    for i, config in enumerate(tqdm(param_space)):
        # Hide prints in this context
        with HiddenPrints():
            result = sys.modules[target].run(**config, show_plots=False)
            results["iterations"].append(result)
        # progress = 100 * i / len(param_space)
        # elapsed = time.time() - search_time
        # print(f'\rprogress: {progress:.2f}% ({elapsed:.0f} s elapsed)\r', end="")
    search_time = time.time() - search_time
    results["total_time"] = search_time
    print(f"progress: finished after {search_time:1f} s")

    return results


def search_wrapper(target, param_space, save_results=True, cache=True):
    # Wrap the search function and save the results if specified

    print(f"performing {len(param_space)} runs with different configurations for {target}...")

    # TODO: somehow check if results will be loaded from cache or not with check_call_in_cache
    # Then add bool value to results["loaded_from_cache"] before returning.
    func = MEM.cache(search) if cache else search
    results = func(target, param_space)
    if save_results:
        pathlib.Path("../results").mkdir(parents=True, exist_ok=True)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file = f"../results/param_search_{target}_{date}.results.json"
        json.dump(results, open(json_file, "w"), indent=4, cls=CustomEncoder)
    
    return results


def optimal_tradeoff(results, method):
    # Determine best hit rate/penetration rate combination 
    # from the available results according to the specified method

    hit_rate_best, pen_rate_best = -1, -1
    run_params = {}
    if method == "youden":
        for r in results["iterations"]:
            pen_rate = r["results"]["pen_rate"]
            hit_rate = r["results"]["hit_rate"]
            if calc.youden_index(hit_rate, pen_rate) > calc.youden_index(hit_rate_best, pen_rate_best):  # maximize
                hit_rate_best = hit_rate
                pen_rate_best = pen_rate
                run_params = copy.deepcopy(r["run_params"])

    elif method == "euklid":
        for r in results["iterations"]:
            pen_rate = r["results"]["pen_rate"]
            hit_rate = r["results"]["hit_rate"]
            if calc.euklidean_dist([hit_rate, pen_rate], [0, 1]) < calc.euklidean_dist([pen_rate_best, hit_rate_best], [0, 1]):  # minmize
                hit_rate_best = hit_rate
                pen_rate_best = pen_rate
                run_params = copy.deepcopy(r["run_params"])
    else:
        raise ValueError(f"Method {method} is invalid!")
    
    # Remove unnecessary parameters from run configuration
    run_params = calc.filter_params(run_params)
    
    return hit_rate_best * 100, pen_rate_best * 100, run_params
          

def main(
          target="lsh",
          show_plots=True,
          save_results=False,
          cache=True,
):
    if target == "lsh":
        # Iterations for each parameter (begin, end, step)
        num_tables_iteration = [1, 5, 10, 15]
        hash_size_iteration = [2]
        param_space = []
        # for num_tables, hash_size in itertools.product(range(*num_tables_iteration), range(*hash_size_iteration)):
        for num_tables in num_tables_iteration:
            for hash_size in hash_size_iteration:
                param_space.append(
                    {
                        "num_tables": num_tables,
                        "hash_size": hash_size,
                    }
                )
        results = search_wrapper(target=target, param_space=param_space, save_results=save_results, cache=cache)
        youden = optimal_tradeoff(results, method="youden")
        euklid = optimal_tradeoff(results, method="euklid")
        print(f"Maximal Youden's index:")
        print("=> hit rate: {:.2f}%, penetration rate: {:.2f}% (parameters: {})".format(*[i for i in youden]))
        print(f"Minimal Euklidean distance to 100% hit rate and 0% penetration rate:")
        print("=> hit rate: {:.2f}%, penetration rate: {:.2f}% (parameters: {})".format(*[i for i in euklid]))
        if show_plots: plot.search_scatter(results)
        if show_plots: plot.lsh_param_search(results)

    elif target == "clustering":
        # Iterations for each parameter (begin, end, step)
        n_clusters_iteration = [1, 2, 3, 1000, 3531]
        param_space = []
        for n_clusters in n_clusters_iteration:
            param_space.append(
                {
                    "n_clusters": n_clusters,
                }
            )
        results = search_wrapper(target=target, param_space=param_space, save_results=save_results, cache=cache)
        youden = optimal_tradeoff(results, method="youden")
        euklid = optimal_tradeoff(results, method="euklid")
        print(f"Maximal Youden's index:")
        print("=> hit rate: {:.2f}%, penetration rate: {:.2f}% (parameters: {})".format(*[i for i in youden]))
        print(f"Minimal Euklidean distance to 100% hit rate and 0% penetration rate:")
        print("=> hit rate: {:.2f}%, penetration rate: {:.2f}% (parameters: {})".format(*[i for i in euklid]))
        if show_plots: plot.search_scatter(results)
        if show_plots: plot.clustering_param_search(results)


if __name__ == '__main__':
    main()
