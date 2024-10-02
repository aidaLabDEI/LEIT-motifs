from base import *
from find_bin_width import *
from stop import stop
import numpy as np, queue, threading, multiprocessing
import numpy.typing as npt
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cProfile
from hash_lsh import RandomProjection
import time

def find_motifs(d, window_size: int, dimensionality: int, r: int, thresh: float, L: int, K: int) -> list:
  
  return