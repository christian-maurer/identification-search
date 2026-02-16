import numpy as np
from utils import read, plot

def main():
    # Load data
    gallery_path = "../data/Gallery"
    probe_path = "../data/Probe"
    gallery = read.Gallery(gallery_path)
    probe = read.Probes(probe_path)
    print(gallery.samples.shape)
    print(probe.samples.shape)

    plot.existing_samples(gallery.sample_ids, probe.sample_ids)


if __name__ == '__main__':
    main()
