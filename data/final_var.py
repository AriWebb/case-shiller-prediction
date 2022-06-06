import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import dataframe_image as dfi

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.base.datetools import dates_from_str

'''
Vector Autoregression Analysis.

Implementation uses info from:
https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
'''

maxlag = 12
CITIES = ['atlanta', 'boston', 'chicago', 'cleveland', 'dallas', 'denver', 'detroit', 'la', 'miami',
          'minneapolis', 'nyc', 'phoenix', 'portland', 'sf', 'seattle', 'tampa', 'dc']


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def adf_test(series, name, verbose, min=.05):
    """Performs ADFuller test to see if time series is stationary.
    Data needs to be stationary to do VAR.
    Args:
        time series of one variable
    Returns:
        True if data is stationary, False otherwise
    """
    m = adfuller(series, autolag="AIC")
    p = m[1]
    if verbose:
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
    if p <= min:
        if verbose:
            print(f"P-Value = {p}. Series is Stationary")
        return True
    else:
        if verbose:
            print(f"P-Value = {p}. Series is Non-Stationary")
        return False


def run_VAR(frame, frame_train, frame_diff, n=12):
    """
    Args:
        frame: original full data frame
        frame_train: training data frame
        frame_diff: time series that has been diffed twice as per convention for this project
        n: number of months to forecast
    Returns:
        final: time series of final forecasted values
    """
    idx = frame_diff.index
    # reformat dates to prep for VAR model
    new = pd.Series(dtype="float64")
    for i, val in enumerate(idx):
        add = ''
        if val[0] == '1':
            add = '1'
        if val[3] == '9':
            new = pd.concat([new, pd.Series('19' + val[3:] + "M" + add + val[1:2], [i])])
        else:
            new = pd.concat([new, pd.Series('20' + val[3:] + "M" + add + val[1:2], [i])])
    new = dates_from_str(new)
    frame_diff.index = pd.DatetimeIndex(new)
    # run VAR model
    model = VAR(frame_diff)
    fitted = model.fit(maxlags=60)
    lag = fitted.k_ar
    f_inp = frame_diff.values[-lag:]
    fc = fitted.forecast(y=f_inp, steps=n)
    fc = pd.DataFrame(fc, index=frame.index[-n:], columns=frame.columns + '_2d')
    # invert transformation back to get real forecast
    rev = fc.copy()
    col = frame_train.columns
    for c in col:
        rev[str(c) + '_1d'] = (frame_train[c].iloc[-1] - frame_train[c].iloc[-2]) + fc[str(c) + '_2d'].cumsum()
        rev[str(c) + '_forecast'] = frame_train[c].iloc[-1] + rev[str(c) + '_1d'].cumsum()
    final = rev.filter(regex='_forecast$', axis=1)
    return final


def diff_data(frame_train, verbose=False):
    """Diffs data twice to make it stationary.
    Args:
         frame_train: a training dataframe
         verbose: whether to print data as performing stationary tests
    Returns:
        frame_diff: the training dataframe diffed twice
    """
    frame_diff = frame_train.copy()
    for i in range(2):
        for n, c in frame_diff.iteritems():
            adf_test(c, c.name, verbose)
        frame_diff = frame_diff.diff().dropna()
    for n, c in frame_diff.iteritems():
        adf_test(c, c.name, verbose)
    return frame_diff


def make_full_plot(frame, path, color="blue"):
    """Makes plots of all data for each city
    Args:
        frame: pandas dataframe
        path: where to output the plot
        color: what color the graph is
    Returns:
        none
    """
    fig, axes = plt.subplots(nrows=5, ncols=3, dpi=120, figsize=(10, 6))
    for i, ax in enumerate(axes.flatten()):
        data = frame[frame.columns[i]]
        ax.plot(data, color=color, linewidth=1)
        ax.set_title(frame.columns[i])
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
        xticks = ax.xaxis.get_major_ticks()
        for i in range(len(xticks)):
            if i % 48 != 0:
                xticks[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_forecast_plot(frame, frame_test, forecasted, city, n=12):
    """Plot forecasted values vs actual values and saves the figure in ./forecasts
    Args:
        frame: original data frame
        frame_test: actual test values
        forecasted: forecasted values
        city: which city this is
        n: number of months
    Returns:
        None
    """
    # plot actuals vs predicted
    fig, axes = plt.subplots(nrows=5, ncols=3, dpi=150, figsize=(10, 10))
    for i, (col, ax) in enumerate(zip(frame.columns, axes.flatten())):
        forecasted[col + '_forecast'].plot(legend=True, ax=ax, color='red').autoscale(axis='x', tight=True)
        frame_test[col][-n:].plot(legend=True, ax=ax, color='blue')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.savefig("./forecasts" + "/" + city)
    plt.close()


def format_data(data_path):
    """Reformat data slightly into a pandas dataframe. Date is the index as a DateTime object rather than a string.
    The first column has each city name, to get a dataframe for only one city, use df[df['city'] == city].

    Args:
        data_path: data csv with full city names
    Returns:
        pandas dataframe formatted correctly for future usage
    """
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    for idx, row in df.iterrows():
        cur = row.name
        if cur[3] == '9':
            cur = cur[0:3] + '19' + cur[3:]
        else:
            cur = cur[0:3] + '20' + cur[3:]
        row.name = pd.to_datetime(cur)
    return df


def split_data(frame, n=12):
    """Splits data into test and train.
    Args:
        frame to split
        number of time periods in train
    Returns:
        train frame
        test frame
    """
    frame_train, frame_test = frame[0:-n], frame[-n:]
    return frame_train, frame_test


def full_model(frame, city, diff_plot=False, forecast_plot=False, n=12):
    """Runs the full model.
    Args:
        frame: data frame to use
        city: city we are operating on
        diff_plot: set to to True to make plots of diffed data
        forecast_plot: set to True to make plots of forecasted values vs test data
        n: number of months to use as test
    Returns:
        forecasted values for last n months
    """
    # grangers causation, see function description
    # c_matrix = grangers_causation_matrix(frame, variables=frame.columns)
    # dfi.export(c_matrix, "./causation_matricies" + "/" + city + ".png")
    # split data into test and train
    frame_train, frame_test = split_data(frame, n)
    # test for data being stationary and diff twice so data is stationary
    frame_diff = diff_data(frame_train)
    # plot diffed data for visualization
    if diff_plot:
        make_full_plot(frame_diff, "./diffed" + "/" + city + ".png", 'green')
    # run VAR model on data
    forecasted = run_VAR(frame, frame_train, frame_diff, 12)
    # plot forecasted vs actual
    if forecast_plot:
        make_forecast_plot(frame, frame_test, forecasted, city)
    return frame_test.copy(), forecasted.copy()


def find_mean_error(df):
    """Finds mean error of predictions 12 months out for all data points in df.
    Args:
         df: dataframe
    Returns:
        mean squared error for every 12 months of each city in the df.
    """
    diffs = []
    for city in CITIES:
        frame = df[df['city'] == city]
        frame = frame.drop(['city'], axis=1)
        for val in range(int(frame.shape[0] * .9), frame.shape[0]):
            test, forecast = full_model(frame.head(val), city)
            diffs.append(np.abs(test['case shiller'][11] - forecast['case shiller_forecast'][11]) / test['case shiller'][11])
    return np.mean(diffs)


def main():
    # first, format the data
    df = format_data("FINAL_all_data_city_names.csv")
    # for city in CITIES:
    #     frame = df[df['city'] == city]
    #     frame = frame.drop(['city'], axis=1)
    #     # to_drop = ['city', 'cpi', 'violent crime', 'property crime', 'patents', 'food and bev']
    #     # for i in to_drop:
    #     #     frame = frame.drop(i, axis=1)
    #     # make plot to visualize data.
    #     make_full_plot(frame, './plots' + "/" + city)
    #     # run model
    #     full_model(frame, city, diff_plot=True, forecast_plot=True)
    print(find_mean_error(df))



if __name__ == '__main__':
    main()
