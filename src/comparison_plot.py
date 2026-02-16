import matplotlib.pyplot as plt
import numpy as np

from utils import calc


plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 20
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def plot_all():

    # 0: baseline
    # 1: Random Projections LSH
    # 2: Clustering
    # 3: Bloom Filter Tree

    # y: penrate
    # x: hitrate
    # s: ident. time

    fig, ax = plt.subplots()
    scale = 2

    # baseline
    ax.scatter(100, 96.09, s=462.76 * scale, c='tab:blue', label="Baseline", alpha=1, edgecolors='none')

    # lsh
    ax.scatter(76.36, 97.79, s=11.08 * scale, c='tab:red', label="Random Projections LSH", alpha=1, edgecolors='none')

    # clustering
    ax.scatter(0.03, 96.09, s=126.83 * scale, c='tab:green', label="Clustering", alpha=1, edgecolors='none')

    # bloom filter
    ax.scatter(0.03, 2.41, s=59.15 * scale, c='tab:orange', label="Bloom Filter Tree", alpha=1, edgecolors='none')

    plt.xlabel("penetration rate (%)")
    plt.ylabel("hit rate (%)")
    plt.legend(frameon=True, facecolor="white")
    plt.show()


plot_all()
