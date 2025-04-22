# <div align = "center"> LEIT-motifs </div>
<div align = "center"> <strong>Scalable discovery of multidimensional motifs in time series</strong>

The algorithm employs Locality Sensitive Hashing `(LSH)` to prune the number of distance computations required to discover subdimensional motifs in multidimensional time series. </div>

## Implementation details
The code is implemented in Python 3.12, and it heavily relies on the 
*numba*, *multiprocessing* and *numpy* libraries to ensure scalability.

`pyproject.toml` is provided to allow the creation of a working environment.

The algorithm has **anytime** properties, computation can be stopped at any given point (i.e., CTRL + C) and the algorithm will return the best results it achieved.

**Discretized Random Projections LSH** has been implemented with the use of *tensoring* to minimize the hash evaluations.

For general use the `LEITmotifs` function does all the heavy lifting (i.e., deals with NaN values, transforms the time series in the correct format for the algorithm, etc.).

We additionally include our implementation of the  *axis-aligned projection algorithm* introduced in the work of *[Minnen et al., 2007](https://faculty.cc.gatech.edu/~isbell/papers/minnen-icdm2007.pdf)*.

## Reproducibility
The scripts in the folder `tests` replicate all the tests performed in the paper and some additional tests that we could
not fit in the paper due to space limitations, the scripts produce *.csv* files in the `results` folder.

The `*_plotter.py` scripts in the folder `results` collect the raw results and plot them.


## Installation as a Python package

```bash
pip install git+https://github.com/aidaLabDEI/LEIT-motifs
```
```python
from LEITmotifs import LEITmotifs
# Example usage
# Find k multidimensional motifs for each dimensionality from 2 to D:
# let Ts be a D-dimensional time series as a numpy array,
# window the length of the motifs to discover,
# and k the number of motifs to discover
motifs, _ = LEITmotifs(Ts, window, k, (2,D))
# Find k multidimensional motifs that span d dimensions, d∈[2,D]:
motifs, _ = LEITmotifs(Ts, window, k, (d,d))
```

## Citing
```bibtex
@article{ceccarello2025leitmotifs,
  title={LEIT-motifs: Scalable Motif Mining in Multidimensional Time Series},
  author={Ceccarello, Matteo and Monaco, Francesco Pio and Silvestri, Francesco},
  journal={arXiv preprint arXiv:2502.14446},
  year={2025}
}
```
