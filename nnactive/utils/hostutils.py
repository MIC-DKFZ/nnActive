import os
from typing import Union


def get_hostname() -> str:
    return os.uname()[1]


def get_verbose(verbose: Union[bool, None]):
    if verbose is not None:
        return verbose
    else:
        hostname = get_hostname()
        cluster_names = ["hdf19", "e230-dg", "lsf22", "e071", "e230-pc31"]
        if [hostname.startswith(hname) for hname in cluster_names]:
            return False
        else:
            return True
