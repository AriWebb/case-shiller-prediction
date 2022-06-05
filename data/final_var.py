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
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    if p <= min:
        if verbose:
            print(f"P-Value = {p}. Series is Stationary")
        return True
    else:
        if verbose:
            print(f"P-Value = {p}. Series is Non-Stationary")
        return False


def make_plot(frame, path):
    """Makes plots of all data for each city and puts them in ./plots folder
    Args:
        pandas dataframe
    Returns:
        none
    """
    # Plot
    fig, axes = plt.subplots(nrows=4, ncols=3, dpi=120, figsize=(10, 6))
    for i, ax in enumerate(axes.flatten()):
        data = frame[frame.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Decorations
        ax.set_title(frame.columns[i])
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(path)


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


def main():
    # first, format the data
    df = format_data("FINAL_all_data_city_names.csv")
    for city in CITIES:
        frame = df[df['city'] == city]
        frame = frame.drop(['city'], axis=1)
        # then, make plot to visualize data.
        path = './plots' + "/" + city
        # make_plot(frame, path)
        # grangers causation, see function description
        # c_matrix = grangers_causation_matrix(frame, variables=frame.columns)
        # dfi.export(c_matrix, "./causation_matricies" + "/" + city + ".png")
        #split data into test and train
        frame_train, frame_test = split_data(frame)
        #test for data being stationary and diffs until it is stationary
        print(city)
        st = False
        iter = 0
        while not st:
            cur = True
            for n, c in frame_train.iteritems():
                cur_st = adf_test(c, c.name, False)
                if not cur_st:
                    cur = False
            st = cur
            frame_train = frame_train.diff().dropna()
            iter += 1
        print(iter)
        make_plot(frame_train, "./diffed.png")
        # run VAR model
        print(frame_train)
        model = VAR(frame_train)
        return







if __name__ == '__main__':
    main()
