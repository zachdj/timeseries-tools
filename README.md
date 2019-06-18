# Time-series Tools
Collection of handy tools for working with time-series data in Pandas dataframes.

## Installation

The tools depend on Pandas and NumPy. [Conda](https://docs.conda.io/en/latest/) can be used to create an environment with 
all dependencies installed.  Just run

```bash
 $ conda env create -f environment.yml
```

Alternatively, dependencies can be installed from the requirements.txt file:

```bash
 $ pip3 install -r requirements.txt
```

## Tools

### Persistence Models

Persistence models are commonly used as baselines for time-series problems.
Persistence models predict future values of the time series assuming that conditions 
remain unchanged between a current or historical observation and the future time.

This library includes three persistence models:

- LatestValuePersistence
    - predicts future values using the latest historical value
- LaggedValuePersistence
    - predicts future values at time T using the closest historical observation 
    to (T - lag)
- SlidingWindowPersistence
    - predicts future values using the mean from a window of historical values.


### Validation Tools

This library also includes a utility function for time-series cross-validation.
Unlike in regular cross-validation, the order of examples in a time series matter.

The `validation.splitters.cross_val_split` function will create a generator that 
yields (train, test) subsets of a pandas DataFrame.  This is similar in spirit to 
[scikit-learn's `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
`TimeSeriesSplit` works well for the simplest case of time-series cross-validation, 
where data points are sorted, evenly spaced, and a cumulative, walk-forward split is desired.  
But `TimeSeriesSplit` is pretty unhelpful in a variety of other cases.  
For example, it cannot handle uneven sampling intervals, leave-one-out cross-validation, 
or imposing limits on the size of training data.
Furthermore, `TimeSeriesSplit` exposes an index-based API designed to work with numpy ndarrays. 
When working with time-series data, it is often more convenient to deal with a time-oriented API 
rather than an index-oriented API.

