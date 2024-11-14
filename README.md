# LEIT-motifs
**Scalable discovery of multidimensional motifs in time series**. 
The algorithm employs `LSH` to prune the number of distance computations required to discover subdimensional motifs in multidimensional time series.

## Implementation detals
`Discretized Random Projections` has been implemented with the use of *tensoring* to minimize the computations.
The implementation makes usage of *shared memory* and *multiprocessing* to ensure scalability and a small memory footprint.

## Extra
The `EvaluationSuite` Jupiter Notebook offer an interactive small evaluation against the SoTA for exact discovery other than many usage examples without the need to download anything. For further evaluation refer to `source/test.py` \
The `Thesis_Plots` notebook ensures repeatability for all the figures included in the thesis. 
