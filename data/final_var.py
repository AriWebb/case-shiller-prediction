import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

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
test = 'ssr_chi2test'


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


def make_plots(df):
    """Makes plots of all data for each city and puts them in ./plots folder
    Args:
        pandas dataframe
    Returns:
        none
    """
    cities = set()
    for i in df['city'].values:
        cities.add(i)
    for city in cities:
        frame = df[df['city'] == city]
        frame = frame.drop(['city'], axis=1)
        # Plot
        fig, axes = plt.subplots(nrows=4, ncols=3, dpi=120, figsize=(10, 6))
        for i, ax in enumerate(axes.flatten()):
            data = frame[frame.columns[i]]
            ax.plot(data, color='red', linewidth=1)
            # Decorations
            ax.set_title(frame.columns[i])
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
        plt.tight_layout()
        plt.savefig('./plots' + city)


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


def main():
    #first, format the data
    df = format_data("FINAL_all_data_city_names.csv")
    # then, make plots to visualize data.
    # make_plots(df)
    # grangers causation, see function description
    c_matrix = grangers_causation_matrix(df, variables=df.columns)
    display(c_matrix)


if __name__ == '__main__':
    main()
