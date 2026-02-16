import numpy as np
import pandas as pd
import pathlib
import time
import joblib
import time

MEM = joblib.Memory(location="cache", verbose=0)


def get_filepaths(path):
    # Return list of files from the directory specified with path

    dir = pathlib.Path(path)
    if not dir.is_dir():
        raise ValueError(f'{dir}" is not a directory')
    filepaths = [pathlib.Path(file) for file in dir.glob("*.npy")]
    if len(filepaths) == 0:
        raise Exception(f"No sample files found in {path}")
    return filepaths


def load_samples(path, n_max=None):
    # Return numpy array of all samples from directory specified with path

    load_time = time.time()
    filepaths = get_filepaths(path)

    first = np.load(filepaths[0])  # derive shape from first sample to preallocate numpy array
    n = len(filepaths) if n_max is None else n_max
    all_samples = np.zeros((n,) + first.shape)
    for i, fp in enumerate(filepaths):
        all_samples[i, :] = np.load(fp)
    
    load_time = time.time() - load_time
    print(f"Preloaded {len(all_samples)} with shape {all_samples[0].shape} from {path} in {load_time:.2f} s")
    return all_samples


def distances_pd(path):
    # Returns the distances table from the textfile as pandas dataframe

    col_names = ['gallery_filename', 'nan', 'probe_filename', 'gallery_idx', 'distance']
    df = pd.read_csv(path, sep=" ", header=None, names=col_names)
    #table = np.loadtxt(path, dtype=str)
    df = df.drop(['nan'], axis=1)
    return df


def identities_pd(path):
    # Returns the identities table from the textfile as pandas dataframe

    col_names = ['any_sample_filename', 'gallery_idx']
    df = pd.read_csv(path, sep=" ", header=None, names=col_names)
    return df


# TODO: If needed inherit from torch (maybe needs transformation then)
#from torch.utils.data import Dataset
#class HcmlDataset(Dataset):
class HcmlDataset():
    def __init__(self, dataset_dir, col_name, cache=True):
        if cache:
            self.__load__ = MEM.cache(self.__load__)
        self.dataset_dir, self.sample_ids, self.samples, self.n = self.__load__(dataset_dir, col_name)
    
    def __load__(self, dataset_dir, col_name):
        d_table = distances_pd("../data/distances.txt")
        sample_ids = d_table.loc[:, col_name]
        # Get unique filenames of dataset while keeping the default order from distances.txt
        sample_ids = sample_ids[np.sort(np.unique(sample_ids, return_index=True)[1])]  # important for probe samples
        n = len(sample_ids)
        first = np.load(pathlib.Path(dataset_dir) / str(sample_ids[0]))  # derive shape from first sample to preallocate numpy array
        samples = np.zeros((n,) + first.shape)
        for i, id in enumerate(sample_ids):
            filepath = pathlib.Path(dataset_dir) / str(id)
            samples[i, :] = np.load(filepath)
        return dataset_dir, sample_ids, samples, n

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def s(self, id):
        # Return sample by sample_id
        # TODO: implement binary search for more efficiency (assumes sorted sample_ids)
        # TODO: maybe use hash table for better search by id?
        for i in self.sample_ids:
            if i == id:
                return self.samples[i]
        print(f"Warning: id {id} not found in {self.__class__}")
        return None  # TODO: rather raise Exception?


class Probes(HcmlDataset):
    def __init__(self, dataset_dir, cache=True):
        super(Probes, self).__init__(dataset_dir, col_name='probe_filename', cache=cache)
class Gallery(HcmlDataset):
    def __init__(self, dataset_dir, cache=True):
        super(Gallery, self).__init__(dataset_dir, col_name='gallery_filename', cache=cache)


def load():
    # Load data
    load_time = time.time()
    gallery = Gallery("../data/Gallery")
    probe = Probes("../data/Probe")
    print(f"n_identities: {gallery.n}")
    print(f"n_probes: {probe.n}")
    print(f"n_features: {gallery.samples.shape[-1]}")
    i_table = identities_pd("../data/identities.txt")
    d_table = distances_pd("../data/distances.txt")
    load_time = time.time() - load_time
    print(f"time to load data: {load_time:.2f} s")

    return gallery, probe, i_table, d_table
