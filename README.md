# LEIT-motifs
**Scalable discovery of multidimensional motifs in time series**.

The algorithm employs `LSH` to prune the number of distance computations required to discover subdimensional motifs in multidimensional time series.

## Implementation details
The code is implemented in Python 3.12, and it heavily relies on the 
*numba*, *multiprocessing* and *numpy* libraries to ensure scalability.

`requirements.txt` is provided to allow the creation of a working environment.

The algorithm has **anytime** properties, computation can be stopped at any given point and the algorithm will return the best results it achieved.

**Discretized Random Projections LSH** has been implemented with the use of *tensoring* to minimize the computations.

For general use the `LEITmotifs` function does all the heavy lifting (i.e., deals with NaN values, transforms the time series in the correct format for the algorithm, etc.).


## Reproducibility
The scripts in the folder `tests` replicate all the tests performed in the paper and some additional tests that we could
not fit in the paper due to space limitations, the scripts produce *.csv* files in the `results` folder.

The file `plotter.py` collects all the raw results and plots them.

Additional tests are described in the file [`Supplementary.pdf`](https://github.com/FrancescoMonaco/LEIT-motifs/blob/main/Supplementary.pdf).
