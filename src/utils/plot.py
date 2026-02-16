import matplotlib.pyplot as plt
import numpy as np

from utils import calc

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 20
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def existing_samples(gallery_ids, probe_ids):
    y_gal = np.zeros(len(gallery_ids))
    y_pro = np.zeros(len(probe_ids))
    plt.scatter(gallery_ids, y_gal, label="gallery samples")
    plt.scatter(probe_ids, y_pro, label="probe samples")
    plt.xlabel("Sample ID")
    plt.legend()
    plt.show()


def hit_vs_pen(hit_rates, pen_rates):
    hit_rates = np.array(hit_rates) * 100
    pen_rates = np.array(pen_rates) * 100
    plt.plot(pen_rates, hit_rates, marker="x", linestyle="dashed")
    plt.xlabel("penetration rate (%)")
    plt.ylabel("hit rate (%)")
    plt.show()


def retrieved_number(y_ret):
    plt.plot(y_ret)
    plt.xlabel("probe")
    plt.ylabel("number of retrieved results")
    plt.show()


def search_scatter(results):
    for r in results["iterations"]:
        pen_rate = r["results"]["pen_rate"] * 100
        hit_rate = r["results"]["hit_rate"] * 100
        run_params = calc.filter_params(r["run_params"])
        label = ", ".join([f'{key}: {value}' for key, value in run_params.items()])
        plt.scatter(pen_rate, hit_rate, label=label)
    plt.xlabel("penetration rate (%)")
    plt.ylabel("hit rate (%)")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()


def explained_variance(explvar, label):
    plt.bar(x=list(range(len(explvar))), height=explvar * 100, label=label)
    plt.plot(np.cumsum(explvar) * 100)
    plt.xlabel("principal components")
    plt.ylabel("explained variance (%)")
    plt.legend(frameon=True)
    plt.show()


def lsh_param_search(results):
    plt.rcParams["font.size"] = 25
    pen_rates = []
    hit_rates = []
    for r in results["iterations"]:
        pen_rate = r["results"]["pen_rate"] * 100
        hit_rate = r["results"]["hit_rate"] * 100
        run_params = calc.filter_params(r["run_params"], blacklist=["seed", "show_plots", "hash_size", "pca"])
        label = ", ".join([f'num_tables = {value}' for key, value in run_params.items()])
        pen_rates.append(pen_rate)
        hit_rates.append(hit_rate)
        plt.scatter(pen_rate, hit_rate, label=label, marker="x", s=200)
    plt.plot(pen_rates, hit_rates, linestyle="dashed")
    plt.xlabel("penetration rate (%)")
    plt.ylabel("hit rate (%)")
    plt.xlim(0, 100)
    plt.ylim(-1, 101)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()


def clustering_param_search(results):
    plt.rcParams["font.size"] = 25
    pen_rates = []
    hit_rates = []
    for r in results["iterations"]:
        pen_rate = r["results"]["pen_rate"] * 100
        hit_rate = r["results"]["hit_rate"] * 100
        run_params = calc.filter_params(r["run_params"])
        label = ", ".join([f'c = {value}' for key, value in run_params.items()])
        pen_rates.append(pen_rate)
        hit_rates.append(hit_rate)
        plt.scatter(pen_rate, hit_rate, label=label, marker="x", s=200)
    plt.plot(pen_rates, hit_rates, linestyle="dashed")
    plt.xlabel("penetration rate (%)")
    plt.ylabel("hit rate (%)")
    plt.xlim(-1, 101)
    plt.ylim(0, 100)
    plt.legend(loc="lower right", fontsize=15)
    plt.show()
