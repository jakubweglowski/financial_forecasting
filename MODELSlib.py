import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from time import time


# SECTION 1: GENERAL DATA METHODS

def choose_arrayMAXorMIN(df: pd.DataFrame, method: str):
    """Choose indexes of a maximum or minimal value from a dataframe

    Args:
        df (pd.DataFrame): dataframe
        method (str): 'max' or 'min'

    Returns:
        tuple: indexes of maximal/minimal element element
    """
    if method not in ['min', 'max']:
        raise ValueError('Argument "method" must be one of "min", "max".')
    if method == 'max':
        ret = np.unravel_index(np.argmax(df, axis=None), df.shape)
    if  method == 'min':
        ret = np.unravel_index(np.argmin(df, axis=None), df.shape)
    return ret

def transform(y: pd.Series):
    """Returns log and diff transformations of given time series y

    Args:
        y (pd.Series): time series to be transformed

    Returns:
        dict:  {'y': y, 
                'logy': np.log(y), 
                'ydiff': y.diff().dropna(), 
                'logydiff': np.log(y).diff().dropna()}
    """
    return {'y': y, 'logy': np.log(y), 'ydiff': y.diff().dropna(), 'logydiff': np.log(y).diff().dropna()}


# SECTION 2: PLOTTING FUNCTIONS

def first_plots(y: pd.Series, tick: str = None):
    """Plots time series,  its logarithm, difference and log-difference values

    Args:
        y (pd.Series): time series
    """
    d = transform(y)
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    d['y'].plot(ax = ax[0, 0])
    ax[0, 0].set(title='$y$', xlabel='Time', ylabel='Value')
    ax[0, 0].grid()
    d['logy'].plot(ax = ax[0, 1])
    ax[0, 1].set(title='$\log(y)$', xlabel='Time', ylabel='Value')
    ax[0, 1].grid()
    d['ydiff'].plot(ax = ax[1, 0])
    ax[1, 0].set(title='$\Delta y$', xlabel='Time', ylabel='Value')
    ax[1, 0].grid()
    d['logydiff'].plot(ax = ax[1, 1])
    ax[1, 1].set(title='$\Delta(\log(y))$', xlabel='Time', ylabel='Value')
    ax[1, 1].grid()
    
    if tick is None:
        fig.suptitle('First plots of modelled variable')
    else:
        fig.suptitle(f'First plots of {tick}')

    plt.show()

def plot_allVARS(df: pd.DataFrame, overlay: bool = False):
    """Plot all of the variables included in data frame

    Args:
        df (pd.DataFrame): data frame (i.e. multivariate time series) with columns to be plotted
    """
    if len(df.columns) == 1:
        fig, ax = plt.subplots(len(df.columns), figsize = (10, 4), layout='constrained')

        for i, y in enumerate(df.columns):
            ax.plot(df[y].dropna(), label=f'{y}')
            ax.set(title=str(df.columns[i]))
            ax.legend()
            ax.grid(b=True)
            
    if len(df.columns) > 1 and overlay:
        title = ''
        h = 4 * len(df.columns)
        fig, ax = plt.subplots(1, figsize = (10, 4), layout='constrained')

        for i, y in enumerate(df.columns):
            title += str(y) + ' and '
            ax.plot(df[y].dropna(), label=f'{y}')
        ax.set(title=title[:-5])
        ax.legend()
        ax.grid(b=True)
            
    if len(df.columns) > 1 and not overlay:
        h = 4 * len(df.columns)
        fig, AX = plt.subplots(len(df.columns), figsize = (10, h), layout='constrained')

        for i, y in enumerate(df.columns):
            ax = AX[i]
            ax.plot(df[y].dropna(), label=f'{y}')
            ax.set(title=str(df.columns[i]))
            ax.legend()
            ax.grid(b=True)
        
    fig.suptitle('VAR analysis: included variables')
    plt.show()

def plot_forecast_univariate(y_train: pd.Series, 
                             y_hat: pd.Series, 
                             y_test: pd.Series = None, 
                             low: pd.Series = None, 
                             high: pd.Series = None,
                             model_name: str = None,
                             tick: str = None):

    fig, ax = plt.subplots(figsize=(10,6), layout='constrained')
    ax.plot(y_train, label='Training Data', marker='o', ms=4)
    ax.plot(y_hat, label='Forecasted Data', color='green', marker='o', ms=4)
    
    if y_test is not None:
        ax.plot(y_test, label='Actual Data', color='orange', marker='o', ms=4)
        
    if low is not None and high is not None:
        ax.fill_between(y_hat.index,
                        low, 
                        high, 
                        color='k', alpha=.15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.legend()
    ax.grid(b=True)
    
    title = f'Forecast'
    if tick is not None:
        title += f' of {tick}'
    if model_name is not None:
        title += f' from {model_name}'
    title += f'\nstart: {y_train.index[0].date()}, end: {y_hat.index[-1].date()}'
    
    fig.suptitle(title)
    fig.show()
    
    return fig, ax
    
def plot_forecast_multivariate(df: pd.DataFrame, 
                               train_index: pd.DatetimeIndex, 
                               test_index: pd.DatetimeIndex,
                               forecast: pd.DataFrame,
                               differenced: dict,
                               conf_int: tuple = None):
    """Function to plot forecasts

    Args:
        df (pd.DataFrame): data frame with 'raw' data
        train_index (pd.DatetimeIndex): indexes of training set
        test_index (pd.DatetimeIndex): indexes of test set
        forecast (pd.DataFrame): data frame with forecast
        conf_int (tuple): data with confidence intervals

    Raises:
        ValueError: informs when df has only one column, this function acts on multivariate
    """
    
    if len(df.columns) == 1:
        raise ValueError('In a VAR model we need more than one variable')
    
    # plot the results
    h = 4 * len(df.columns)
    fig, ax = plt.subplots(len(df.columns), figsize=(10, h), layout='constrained')

    for i, y in enumerate(df.columns):
        ax[i].plot(df.loc[train_index, y], label='Training data')
        ax[i].plot(df.loc[test_index, y], label='Test data')
        ax[i].plot(forecast[y], label='Forecast')
        ax[i].legend()
        ax[i].set(title=y)
        ax[i].grid()
        if conf_int != None:
            low, high = conf_int
            if differenced[y]:
                # time series y has been differenced to enforce stationarity
                # we have to cumsum() the forecast and add the previous value
                low[y] = np.cumsum(low[y]) + df.loc[train_index][y][-1]
                high[y] = np.cumsum(high[y]) + df.loc[train_index][y][-1]
            ax[i].fill_between(test_index, 
                               low[y], 
                               high[y],
                               color="grey", alpha=.6)
        
    fig.suptitle('Results of VAR forecast')
    plt.show()
    


# SECTION 3: TESTING STATIONARITY

def plotacfpacf(y: pd.Series):
    """Plot ACF and PACF  of a univariate time series

    Args:
        y (pd.Series): univariate time series of data
    """
    fig, ax = plt.subplots(2, figsize=(10, 6), layout='constrained')

    plot_acf(y, ax = ax[0])
    plot_pacf(y, ax = ax[1])

    plt.show()
    
def performADF(y: pd.Series, verbose: bool = False) -> float:
    """Perform Augmented Dickey-Fuller stationarity test.
    Null hypothesis: series is not stationary.

    Args:
        y (pd.Series): time series to test stationarity
        verbose (bool, optional): print details of ADF test. Defaults to False.

    Returns:
        float: p-value from ADF
    """
    dftest = adfuller(y, autolag = 'AIC')
    if verbose:
        print("1. ADF : ", dftest[0])
        print("2. P-Value : ", dftest[1])
        print("3. Num Of Lags : ", dftest[2])
        print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
        print("5. Critical Values :")
        for key, val in dftest[4].items():
            print("\t",key, ": ", val)
    return dftest[1]

def adfuller_allVARS(df: pd.DataFrame, 
                     enforce_stationarity: bool = True, 
                     verbose: bool = False) -> tuple:
    """Perform ADF on all columns of a data frame, make them stationary if told to.

    Args:
        df (pd.DataFrame): data frame with time series as columns (i.e. multivariate time series)
        enforce_stationarity (bool, optional): whether to perform differencing to enforce stationarity. Defaults to True.
        verbose (bool, optional): print information from the test
    Returns:
        tuple: (stationarized data frame, dict with info which columns had been differenced)
    """
    data = df.copy()
    
    which_vars_differenced = {}
    for y in data.columns:
        
        which_vars_differenced[y] = False
        
        if verbose:
            print(f'Testing time series {y}:')
        p = performADF(data[y])
        if verbose:
            print(f'\tAugmented Dickey-Fuller test p-value = {p:.4f}')
        
        if p > 0.01:
            if verbose and enforce_stationarity:
                print(f'\t...Time series {y} is not stationary and hence will be differenced.')
            if enforce_stationarity:
                data[y] = data[y].diff()
                which_vars_differenced[y] = True
            
            
    return data.dropna(), which_vars_differenced


# SECTION 4: ARIMA model

def arima_outoftimeCV(y: pd.Series, 
                      max_p: int,
                      max_q: int, 
                      d: int = 1,
                      n_splits: int = 5,
                      max_train_size: int = 360,
                      test_size: int = 5,
                      verbose: bool = False):
    """Perform Out-Of-Time (OOT) Cross Validation for ARIMA model selection using AICc as the criterion of selection.

    Args:
        y (pd.Series): Time series of data, required stationarity
        max_p (int): Max. order of autoregression
        max_q (int): Max. order of moving average
        d (int): Order of differencing. If y is stationary, set d = 0. Defaults to 1.
        n_splits (int, optional): Number of CV folds. Defaults to 5.
        max_train_size (int, optional): Max. obs in training set. Defaults to 360.
        test_size (int, optional): Max. obs in test set. Defaults to 5.
        verbose (bool, optional):  Print detailed output. Defaults to False.

    Returns:
        dict: dictionary containing dataframes with summary of analysis
    """
    startingtime = time()
    
    # DataFrame with results of analysis, will be returned at the end
    CVresults = {'RMSE': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1))),
                 'AllDays': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1))),
                 'LastDay': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1))),
                 'FirstDay': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1))),
                 'AIC': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1))),
                 'BIC': pd.DataFrame(np.zeros(shape=(max_p+1, max_q+1)))
                 }
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            
            tic = time()
            
            if p == 0 and q == 0:
                # ARIMA(0, 0, 0) is a bad predictive model
                CVresults['RMSE'].iloc[p, q] = np.inf
                CVresults['AllDays'].iloc[p, q] = 0
                CVresults['LastDay'].iloc[p, q] =  0
                CVresults['FirstDay'].iloc[p, q] =  0
                CVresults['AIC'].iloc[p, q] = np.inf
                CVresults['BIC'].iloc[p, q] = np.inf
                continue
            
            # set ARIMA model order
            order = (p, d, q)
            
            if verbose:
                print(f'ARIMA({p},{d},{q})')
                
            # initialize array to remember RMSE's
            RMSEs = np.zeros(n_splits)
            AllDays = np.zeros(n_splits)
            LastDays = np.zeros(n_splits)
            FirstDays = np.zeros(n_splits)
            AICs = np.zeros(n_splits)
            BICs = np.zeros(n_splits)
            
            kf = TimeSeriesSplit(n_splits=n_splits,
                                 max_train_size=max_train_size,
                                 test_size=test_size).split(y)
            
            # proper CV
            for i, (train_index, test_index) in enumerate(kf):
                    
                train_index = y.index[train_index]
                test_index = y.index[test_index]
                if (i == 0 or i % 10 == 9) and verbose:
                    print(f'\tFold: {i+1} / {n_splits}')
                    print(f'\t\tObs. in train set: {len(train_index)}, start={train_index[0].date()}, end={train_index[-1].date()}')
                    print(f'\t\tObs. in test set: {len(test_index)}, start={test_index[0].date()}, end={test_index[-1].date()}')
                
                try:
                    model = ARIMA(y[train_index], order=order)
                    fitted = model.fit()
                
                    forecast = fitted.get_forecast(steps=len(test_index))
                    yhat = forecast.predicted_mean
                    yhat.index = test_index
                    
                    # Check whether a direction of forecast was correct
                    lastval = y[train_index[-1]]
                    AllDays[i] = (np.mean(yhat - lastval)*np.mean(y[test_index] - lastval) > 0)
                    LastDays[i] = (1 if (yhat[test_index[-1]] - lastval)*(y[test_index[-1]] - lastval) > 0 else 0)
                    FirstDays[i] = (1 if (yhat[test_index[0]] - lastval)*(y[test_index[0]] - lastval) > 0 else 0)
                    
                    # Save information criteria
                    AICs[i] = fitted.aic
                    BICs[i] = fitted.bic
                    
                    # Calculate RMSE of the forecast
                    RMSEs[i] = mean_squared_error(y[test_index], yhat)**0.5                             
                
                except np.linalg.LinAlgError as e:
                    print(f'\t\t{e}')
                    print('\t\tCV proceeds...')
                    # LU decomposition error
            
            if verbose:
                print(f'...Time: {time() - tic :.2f} seconds.\n') 
                   
            CVresults['RMSE'].iloc[p, q] = np.mean(RMSEs)
            CVresults['AllDays'].iloc[p, q] = np.mean(AllDays)
            CVresults['LastDay'].iloc[p, q] =  np.mean(LastDays)
            CVresults['FirstDay'].iloc[p, q] =  np.mean(FirstDays)
            CVresults['AIC'].iloc[p, q] = np.mean(AICs)
            CVresults['BIC'].iloc[p, q] = np.mean(BICs)
    
    if verbose:
        print(f'Overall time of computations: {(time() - startingtime)/60:.2f} minutes.') 
        
    return CVresults

def choose_ARIMAorder(results: dict):
    
    ret_dict = {'RMSE': choose_arrayMAXorMIN(results['RMSE'], method='min'),
                'AllDays': choose_arrayMAXorMIN(results['AllDays'], method='max'),
                'FirstDay': choose_arrayMAXorMIN(results['FirstDay'], method='max'),
                'LastDay': choose_arrayMAXorMIN(results['LastDay'], method='max'),
                'AIC': choose_arrayMAXorMIN(results['AIC'], method='min'),
                'BIC': choose_arrayMAXorMIN(results['BIC'], method='min')
                }
    
    ret_vals = {'AllDays': np.max(np.max(results['AllDays'])),
                'FirstDay': np.max(np.max(results['FirstDay'])),
                'LastDay': np.max(np.max(results['LastDay']))}

    return ret_dict, ret_vals


# SECTION 5: VAR model

def VARmodel(df: pd.DataFrame,
              lags: int, 
              obs_in_train_set: int,
              enforce_stationarity: bool = True,
              steps: int = -1):
    
    if len(df.columns) == 1:
        raise ValueError('In a VAR model we need more than one variable')
    if steps <= 0 or steps > lags:
        # prediction from VAR model should not range longer than the number of lagged variables
        steps = lags
    
    # retrieve info about stationarity, enforce it if told to
    df_stationary, differenced = adfuller_allVARS(df, enforce_stationarity, verbose = False)
    
    train_index = df_stationary.index[-steps-obs_in_train_set:-steps]
    df_train = df_stationary.loc[train_index]

    test_index = df_stationary.index[-steps:]
    df_test = df_stationary.loc[test_index]

    # initialize the model
    model = VAR(df_train)

    # determine order of the model
    results = model.fit(lags)
    
    forecast_interval = results.forecast_interval(df_train.values[-lags:], steps, alpha = 0.5)
    
    # get forecast
    forecast = pd.DataFrame(forecast_interval[0],
                            index = test_index,
                            columns = df_stationary.columns)
    
    # 90% confidence intervals
    low = pd.DataFrame(forecast_interval[1],
                            index = test_index,
                            columns = df_stationary.columns)
    
    high = pd.DataFrame(forecast_interval[2],
                            index = test_index,
                            columns = df_stationary.columns)

    # re-difference if needed
    for y in forecast.columns:
        if differenced[y]:
            # time series y has been differenced to enforce stationarity
            # we have to cumsum() the forecast and add the previous value
            forecast[y] = np.cumsum(forecast[y]) + df.loc[train_index][y][-1]
            
    plot_forecast_multivariate(df, train_index, test_index, forecast, differenced, (low, high))