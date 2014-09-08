"""Dataset helper for online downloads."""
import os
import zipfile
from scipy.io import loadmat
from .utils import download

TENSORLIB_DATASETS_DIR = os.path.expanduser("~/tensorlib_data")

# Sites:
# http://three-mode.leidenuniv.nl/
# http://www.sci.utah.edu/~gk/DTI-data/
# http://perception.i2r.a-star.edu.sg/bk_model/bk_index.html
# http://www.birncommunity.org/category/data-catalog/
# http://www.cs.toronto.edu/~dross/ivt/
# Hard to download...
# http://www.models.life.ku.dk/datasets
# http://www.models.life.ku.dk/nwaydata


def load_bread():
    """
    Load brod.mat dataset originally from http://www.models.life.ku.dk/datasets .
    """
    module_path = os.path.join(os.path.dirname(__file__), "data")
    with open(os.path.join(module_path, "bread.txt")) as f:
        descr = f.read()
    matfile = os.path.join(module_path, "brod.mat")
    d = loadmat(matfile)
    X = d['X'].reshape(d['DimX'].ravel())
    meta = {k: d[k] for k in d.keys() if k not in ['X', 'DimX']}
    meta['DESC'] = descr
    return X, meta


def fetch_decmeg():
    """
    Get and load a subject (subject 4) from the DECMEG dataset.
    Currently, this data is housed on the public Dropbox of one of the authors,
    until a permanent solution to download from the main repository is found.
    """
    subdir = "decmeg"
    data_fname = "train_subject04.mat"
    full_path = os.path.join(TENSORLIB_DATASETS_DIR, subdir)
    data_file = os.path.join(full_path, data_fname)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    if not os.path.exists(data_file):
        local_fname = "tmp.zip"
        tmp_file = os.path.join(full_path, local_fname)
        link = "https://dl.dropboxusercontent.com/u/15378192/DECMEG.zip"
        server_fname = "DECMEG.zip"
        download(link, server_fname, tmp_file)
        z = zipfile.ZipFile(tmp_file)
        z.extractall(path=full_path)
        os.remove(tmp_file)

    matfile = os.path.join(full_path, data_fname)
    d = loadmat(matfile)
    X = d['X']
    meta = {k: d[k] for k in d.keys() if k not in ['X']}
    return X, meta
