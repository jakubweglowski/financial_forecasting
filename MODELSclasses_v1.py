import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf#, acf, pacf
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.seasonal import STL, MSTL
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse

from dataloader import getXTB
from MODELSlib import *

from warnings import filterwarnings as flt
flt('ignore')

################################################################################                   
# Functions assessing forecasts
def assess_forecast(lastval, forecast: pd.Series, y_test: pd.Series):
    return (1 if np.mean(forecast - lastval)*np.mean(y_test - lastval) > 0 else 0)

def assess_direction(forecast: pd.Series, y_test: pd.Series):
    return (1 if (forecast[-1] - forecast[0])*(y_test[-1] - y_test[0]) > 0 else 0)

def assess_firstday(lastval, forecast: pd.Series, y_test: pd.Series):
    return (1 if (forecast[0] - lastval)*(y_test[0] - lastval) > 0 else 0)
################################################################################


################################################################################                   
class SeasonalAnalysis:
    
    def __init__(self, y: pd.Series):
        self.y = y
        
    def plotSTL(self, period: int = 7, tick: str = None):
        res = STL(self.y, period = period).fit()

        fig, AX = plt.subplots(3, figsize=(10, 12), layout='constrained')

        ax = AX[0]
        ax.plot(self.y, label='True data')
        ax.plot(res.trend, label='Trend')
        ax.legend()
        ax.grid(b=True)

        ax = AX[1]
        ax.plot(res.seasonal, label='Seasonal')
        ax.legend()
        ax.grid(b=True)

        ax = AX[2]
        ax.plot(res.resid, label='Residuals')
        ax.legend()
        ax.grid(b=True)

        if tick is None:
            fig.suptitle('Seasonal decomposition using STL')
        else:
            fig.suptitle(f'Seasonal decomposition of {tick} using STL')
        fig.show()
        
    def plotMSTL(self, periods: tuple = (7, 30), tick: str = None):
        res = MSTL(self.y, periods = periods).fit()

        fig, AX = plt.subplots(3, figsize=(10, 12), layout='constrained')

        ax = AX[0]
        ax.plot(self.y, label='True data')
        ax.plot(res.trend, label='Trend')
        ax.legend()
        ax.grid(b=True)

        ax = AX[1]
        ax.plot(res.seasonal.iloc[:, 0], label='Weekly')
        ax.plot(res.seasonal.iloc[:, 1], label='Monthly')
        ax.legend()
        ax.grid(b=True)

        ax = AX[2]
        ax.plot(res.resid, label='Residuals')
        ax.legend()
        ax.grid(b=True)

        if tick is None:
            fig.suptitle('Seasonal decomposition using MSTL')
        else:
            fig.suptitle(f'Seasonal decomposition of {tick} using MSTL')

        fig.show()
    
    def decompose(self, method: str = 'stl', period: int | tuple = 7) -> dict:
        if method == 'stl':
            res = STL(self.y, period = period).fit()
        if method == 'mstl':
            res = MSTL(self.y, periods = period).fit()
        return {'trend': res.trend, 'seasonal': res.seasonal,  'residual': res.resid}


################################################################################
class Univariate:
    
    def __init__(self,
                 model: str, #'ARIMA', 'STL_with_ARIMA' or 'TREND_SEASONAL' -- other in progress
                 y: pd.Series,
                 max_train_size: int,
                 test_size: int,
                 method: str = None,
                 period: int | tuple = None,
                 tick: str = None):
        self.model = model
        
        self.y = y
        self.index = self.y.index
        
        self.method = method # use only if model == 'TREND_SEASONAL'
        self.period = period # use only if model == 'STL' or 'TREND_SEASONAL'
        
        print('Num. obs: ', len(self.index))
        
        self.p, self.d, self.q = (0, 0, 0)
        self.max_train_size = max_train_size
        self.test_size = test_size
        
        # forecast assessments
        self.max_pred = -1.0       
        self.max_dir = -1.0
        self.max_firstday = -1.0
        self.min_rmse = np.inf
        self.min_aic = np.inf
        
        # confidence level
        self.alpha = 0.01
        
        # name of analyzed instrument
        self.tick = tick
        if tick is not None:
            print(f'Analysis of {tick}')
        
        if model == 'TREND_SEASONAL':
            if method == 'stl':
                dec = SeasonalAnalysis(self.y).decompose('stl', period)
                self.trend = dec['trend']
                self.seas = dec['seasonal']
            if method == 'mstl':
                dec = SeasonalAnalysis(self.y).decompose('mstl', period)
                self.trend = dec['trend']
                self.seas1 = dec['seasonal'].iloc[:, 0]
                self.seas2 = dec['seasonal'].iloc[:, 1]
    
    def first_plots(self):
        fig, ax = plt.subplots(4, figsize=(10, 16), layout='constrained')
        
        self.y.plot(ax = ax[0])
        ax[0].set(title='$y$', xlabel='Time', ylabel='Value')
        ax[0].grid()
        
        np.log(self.y).plot(ax = ax[1])
        ax[1].set(title='$\log(y)$', xlabel='Time', ylabel='Value')
        ax[1].grid()
        
        self.y.diff().dropna().plot(ax = ax[2])
        ax[2].set(title='$\Delta y$', xlabel='Time', ylabel='Value')
        ax[2].grid()
        
        np.log(self.y).diff().dropna().plot(ax = ax[3])
        ax[3].set(title='$\Delta(\log(y))$', xlabel='Time', ylabel='Value')
        ax[3].grid()
        if self.tick is None:
            fig.suptitle('First plots', fontsize=22)
        else:
            fig.suptitle(f'First plots of {self.tick}', fontsize=22)
            
        fig.show()
        
        return fig, ax
    
    def trainCV(self, 
                 max_p: int = 4, 
                 max_q: int = 4,
                 criterion: str = 'pred',
                 verbose: bool = True,
                 plot: bool = False):
        
        TIC = time()
        if self.model == 'TREND_SEASONAL':
            self.d = (0 if adfuller(self.trend, maxlag=40)[1] < 0.01 else 1)
        else: # model == 'ARIMA' or 'STL_with_ARIMA'
            self.d = (0 if adfuller(self.y, maxlag=40)[1] < 0.01 else 1)

        br = False # boolean to exit outer loop if forecast stats are good enough
        for p in range(max_p + 1):
            if br:
                break
            for q in range(max_q + 1):
                
                tic = time()
                
                # model name section
                if self.model == 'TREND_SEASONAL':
                    model_name = (f'TRSE({p},{self.d},{q})' if self.method == 'stl' else f'MSTL with ARIMA({p},{self.d},{q})')
                if self.model == 'ARIMA':
                    model_name = f'ARIMA({p},{self.d},{q})'
                if self.model == 'STL_with_ARIMA':
                    model_name = f'STL with ARIMA({p},{self.d},{q})'
                if verbose:
                    print(model_name)
                
                # Formula is as follows
                # splits = (len(y.index) - (train+test) + 1)//test + 1
                splits = (len(self.index) - (self.max_train_size+self.test_size) + 1)//self.test_size + 1
                kf = TimeSeriesSplit(n_splits=splits,
                                 max_train_size=self.max_train_size,
                                 test_size=self.test_size).split(self.y)
                
                # arrays for forecast assessments
                first_day = np.zeros(splits)
                pred = np.zeros(splits)
                dir = np.zeros(splits)
                rmse = np.zeros(splits)
                if self.model == 'ARIMA':
                    aic = np.zeros(splits)
                
                # which plot to include
                rand = np.random.randint(0, splits)
                
                # proper CV
                for i, (train_index, test_index) in enumerate(kf):
                        
                    train_index = self.index[train_index]
                    test_index = self.index[test_index]
                    
                    if (i == 0 or i % 10 == 9) and verbose:
                        print(f'\tFold: {i+1} / {splits}')
                        print(f'\t\tObs. in train set: {len(train_index)}, start: {train_index[0].date()}, end: {train_index[-1].date()}')
                        print(f'\t\tObs. in test set: {len(test_index)}, start: {test_index[0].date()}, end: {test_index[-1].date()}')
                    
                    try:
                        if self.model == 'ARIMA':
                            model = ARIMA(self.y[train_index], 
                                          order = (p, self.d, q))
                            fitted = model.fit()
                            aic[i] = fitted.aic
                            forecast = fitted.get_forecast(steps=len(test_index))
                            yhat = forecast.predicted_mean
                            low = forecast.conf_int(alpha = self.alpha).iloc[:, 0]
                            high = forecast.conf_int(alpha = self.alpha).iloc[:, 1]
                            yhat.index = low.index = high.index = test_index
                            
                        if self.model == 'STL_with_ARIMA':
                            model = STLForecast(self.y[train_index], 
                                                ARIMA, 
                                                model_kwargs={'order': (p, self.d, q)}, 
                                                period = self.period)
                            fitted = model.fit()
                        
                            yhat = fitted.forecast(steps=len(test_index))
                            yhat.index = test_index
                            
                        if self.model == 'TREND_SEASONAL':
                            # 1. Forecasting 'trend' component
                            # variant: model_tr = STLForecast(self.trend[train_index], ARIMA, model_kwargs={'order': (p, self.d, q)}, period = self.period)
                            model_tr = ARIMA(self.trend[train_index], 
                                             order = (p, self.d, q))
                            fitted_tr = model_tr.fit()
                            forecast_tr = fitted_tr.get_forecast(steps=len(test_index))
                            yhat_tr = forecast_tr.predicted_mean
                            low = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 0]
                            high = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 1]
                            
                            # 2. Forecasting 'seasonal' component
                            if self.method == 'stl':
                                model_seas = STLForecast(self.seas[train_index], ARIMA, model_kwargs={'order': (p, 0, q)}, period = self.period)
                                fitted_seas = model_seas.fit()
                                yhat_seas = fitted_seas.forecast(steps=len(test_index))
                                
                            if self.method == 'mstl':
                                model_seas1 = STLForecast(self.seas1[train_index], ARIMA, model_kwargs={'order': (p, 0, q)}, period = self.period[0])
                                fitted_seas1 = model_seas1.fit()
                                yhat_seas1 = fitted_seas1.forecast(steps=len(test_index))
                                
                                model_seas2 = STLForecast(self.seas2[train_index], ARIMA, model_kwargs={'order': (p, 0, q)}, period = self.period[1])
                                fitted_seas2 = model_seas2.fit()
                                yhat_seas2 = fitted_seas2.forecast(steps=len(test_index))
                                
                                yhat_seas = yhat_seas1 + yhat_seas2
                                
                            # 3. Combining the two
                            yhat = yhat_seas + yhat_tr
                            yhat.index = low.index = high.index = test_index
                            
                        # forecast analysis
                        first_day[i] = assess_firstday(lastval = self.y[train_index][-1], forecast = yhat, y_test = self.y[test_index])
                        pred[i] = assess_forecast(lastval = self.y[train_index][-1], forecast = yhat, y_test = self.y[test_index])
                        dir[i] = assess_direction(forecast = yhat, y_test = self.y[test_index])
                        rmse[i] = np.sqrt(mse(y_true=self.y[test_index], y_pred=yhat))
                        
                        if i in [rand, rand+10] and max(p, q) > 0 and plot:
                            try:
                                plot_forecast_univariate(y_train = self.y[train_index],
                                                        y_hat = yhat,
                                                        y_test = self.y[test_index],
                                                        low = low,
                                                        high = high,
                                                        model_name = model_name,
                                                        tick = self.tick)
                            except:
                                plot_forecast_univariate(y_train = self.y[train_index],
                                                        y_hat = yhat,
                                                        y_test = self.y[test_index],
                                                        model_name = model_name,
                                                        tick = self.tick)
                        
                    except np.linalg.LinAlgError as e:
                        # LU decomposition error
                        print(f'\t\t{e}')
                        print('\t\tCV proceeds...')
                            
                if np.mean(pred) > self.max_pred:
                    self.max_pred = np.mean(pred)
                    if verbose:
                        print(f'...MAX PRED! Assessment: {self.max_pred=:.3f}')
                    if criterion == 'pred':
                        self.p, self.q = p, q
                        if self.max_pred > 0.99:
                            br = True
                            break
                
                if np.mean(dir) > self.max_dir:
                    self.max_dir = np.mean(dir)
                    if verbose:
                        print(f'...MAX DIR! Assessment: {self.max_dir=:.3f}')
                    if criterion == 'dir':
                        self.p, self.q = p, q
                        if self.max_dir > 0.99:
                            br = True
                            break
                        
                if np.mean(first_day) > self.max_firstday:
                    self.max_firstday = np.mean(first_day)
                    if verbose:
                        print(f'...MAX FIRST_DAY! Assessment: {self.max_firstday=:.3f}')
                    if criterion == 'firstday':
                        self.p, self.q = p, q
                        if self.max_firstday > 0.99:
                            br = True
                            break
                        
                if np.mean(rmse) < self.min_rmse:
                    self.min_rmse = np.mean(rmse)
                    if verbose:
                        print(f'...MIN RMSE! Assessment: {self.min_rmse=:.3f}')
                    if criterion == 'rmse':
                        self.p, self.q = p, q
                        
                if self.model == 'ARIMA':
                    if np.mean(aic) < self.min_aic:
                        self.min_aic = np.mean(aic)
                        if verbose:
                            print(f'...MIN AIC! Assessment: {self.min_aic=:.3f}')
                        if criterion == 'aic':
                            self.p, self.q = p, q
                    
                if verbose:
                    print(f'\t...Time: {time() - tic :.2f} seconds.\n')
        if verbose:
            print('Optimal parameters:')
            print(f'p: {self.p}, d: {self.d}, q: {self.q}')
            print('\nForecast statistics:')
            print(f'\tMean: {self.max_pred:.3f}')
            print(f'\tDirection: {self.max_dir:.3f}')
            print(f'\tFirst day: {self.max_firstday:.3f}')
            print(f'\tRMSE: {self.min_rmse:.3f}')
            if self.model == 'ARIMA':
                print(f'\tAIC: {self.min_aic:.3f}')
            print(f'\n...Total time: {(time() - TIC) / 60 :.2f} minutes.\n')
    
    def CV_winner(self, 
                    plot: int | None = 10,
                    return_stats: bool = False, 
                    verbose: bool = True):
        
        tic = time()
                
        # model name section
        if self.model == 'TREND_SEASONAL':
            model_name = (f'TRSE({self.p},{self.d},{self.q})' if self.method == 'stl' else f'MSTL with ARIMA({self.p},{self.d},{self.q})')
        if self.model == 'ARIMA':
            model_name = f'ARIMA({self.p},{self.d},{self.q})'
        if self.model == 'STL_with_ARIMA':
            model_name = f'STL with ARIMA({self.p},{self.d},{self.q})'
        if verbose:
            print(f'Analyzing winner: {model_name}')
        
        # (len(y.index) - (train+test) + 1)//test + 1
        splits = (len(self.index) - (self.max_train_size+self.test_size) + 1)//self.test_size + 1
        kf = TimeSeriesSplit(n_splits = splits,
                            max_train_size = self.max_train_size,
                            test_size = self.test_size).split(self.y)
        
        first_day = np.zeros(splits)
        pred = np.zeros(splits)
        dir = np.zeros(splits)
        rmse = np.zeros(splits)
        if self.model == 'ARIMA':
            aic = np.zeros(splits)
        
        # proper CV
        for i, (train_index, test_index) in enumerate(kf):
                
            train_index = self.index[train_index]
            test_index = self.index[test_index]
            
            if (i == 0 or i % 10 == 9) and verbose:
                print(f'\tFold: {i+1} / {splits}')
                print(f'\t\tObs. in train set: {len(train_index)}, start: {train_index[0].date()}, end: {train_index[-1].date()}')
                print(f'\t\tObs. in test set: {len(test_index)}, start: {test_index[0].date()}, end: {test_index[-1].date()}')
            
            try:
                if self.model == 'ARIMA':
                    model = ARIMA(self.y[train_index], 
                                    order = (self.p, self.d, self.q))
                    fitted = model.fit()
                    aic[i] = fitted.aic
                
                    forecast = fitted.get_forecast(steps=len(test_index))
                    yhat = forecast.predicted_mean
                    low = forecast.conf_int(alpha = self.alpha).iloc[:, 0]
                    high = forecast.conf_int(alpha = self.alpha).iloc[:, 1]
                    yhat.index = low.index = high.index = test_index
                    
                if self.model == 'STL_with_ARIMA':
                    model = STLForecast(self.y[train_index], 
                                        ARIMA, 
                                        model_kwargs = {'order': (self.p, self.d, self.q)}, 
                                        period = self.period)
                    fitted = model.fit()
                
                    yhat = fitted.forecast(steps=len(test_index))
                    yhat.index = test_index
                    
                if self.model == 'TREND_SEASONAL':
                    # 1. Forecasting 'trend' component
                    # variant: model_tr = STLForecast(self.trend[train_index], ARIMA, model_kwargs={'order': (p, self.d, q)}, period = self.period)
                    model_tr = ARIMA(self.trend[train_index], 
                                        order = (self.p, self.d, self.q))
                    fitted_tr = model_tr.fit()
                    forecast_tr = fitted_tr.get_forecast(steps=len(test_index))
                    yhat_tr = forecast_tr.predicted_mean
                    low = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 0]
                    high = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 1]
                    
                    # 2. Forecasting 'seasonal' component
                    if self.method == 'stl':
                        model_seas = STLForecast(self.seas[train_index], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period)
                        fitted_seas = model_seas.fit()
                        yhat_seas = fitted_seas.forecast(steps=len(test_index))
                        
                    if self.method == 'mstl':
                        model_seas1 = STLForecast(self.seas1[train_index], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period[0])
                        fitted_seas1 = model_seas1.fit()
                        yhat_seas1 = fitted_seas1.forecast(steps=len(test_index))
                        
                        model_seas2 = STLForecast(self.seas2[train_index], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period[1])
                        fitted_seas2 = model_seas2.fit()
                        yhat_seas2 = fitted_seas2.forecast(steps=len(test_index))
                        
                        yhat_seas = yhat_seas1 + yhat_seas2
                        
                    # 3. Combining the two
                    yhat = yhat_seas + yhat_tr
                    yhat.index = low.index = high.index = test_index
                    
                # forecast analysis
                first_day[i] = assess_firstday(lastval = self.y[train_index][-1], forecast = yhat, y_test = self.y[test_index])
                pred[i] = assess_forecast(lastval = self.y[train_index][-1], forecast = yhat, y_test = self.y[test_index])
                dir[i] = assess_direction(forecast = yhat, y_test = self.y[test_index])
                rmse[i] = np.sqrt(mse(y_true=self.y[test_index], y_pred=yhat))
                    
                if plot is not None:
                    if i % plot in [0, 1, 2]:
                        ret = (np.mean(yhat) - self.y[train_index][-1])/self.y[train_index][-1]
                        decision = ('Wait' if ret <= 0.01 and ret >= -0.01 else ('Buy' if ret > 0.01 else 'Sell'))
                        try:
                            fig = plot_forecast_univariate(y_train = self.y[train_index],
                                                    y_hat = yhat,
                                                    y_test = self.y[test_index],
                                                    low = low,
                                                    high = high,
                                                    model_name = model_name,
                                                    tick = self.tick)[0]
                            fig.text(x = 0.002, y = 0.93, 
                                        s=f'Decision: {decision}\nRate: {ret*100:.2f}%\nTime: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                                        bbox=dict(facecolor='none', edgecolor='black'),
                                        fontsize=8.5)
                        except:
                            fig = plot_forecast_univariate(y_train = self.y[train_index],
                                                    y_hat = yhat,
                                                    y_test = self.y[test_index],
                                                    model_name = model_name,
                                                    tick = self.tick)[0]
                            fig.text(x = 0.002, y = 0.93, 
                                        s=f'Decision: {decision}\nRate: {ret*100:.2f}%\nTime: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                                        bbox=dict(facecolor='none', edgecolor='black'),
                                        fontsize=8.5)
                
            except np.linalg.LinAlgError as e:
                # LU decomposition error
                print(f'\t\t{e}')
                print('\t\tCV proceeds...')
        
        stats = {'mean': np.mean(pred),
                 'dir': np.mean(dir),
                 'first_day': np.mean(first_day),
                 'rmse': np.mean(rmse)}
        
        if verbose:
            print('\nForecast statistics:')
            print(f'\tMean: {stats["mean"]:.3f}')
            print(f'\tDirection: {stats["dir"]:.3f}')
            print(f'\tFirst day: {stats["first_day"]:.3f}')
            print(f'\tRMSE: {stats["rmse"]:.3f}')
            if self.model == 'ARIMA':
                print(f'\tAIC: {self.min_aic:.3f}')
            
            print(f'\n...Total time: {(time() - tic) / 60 :.2f} minutes.\n')
            
        if return_stats == True:
            return stats
    
    def predict(self, verbose: bool = True):
        
        # how many days in advance to forecast
        forecast_range = self.test_size
        start = datetime.strptime(self.y.index[-1].strftime('%Y-%m-%d'), '%Y-%m-%d') + timedelta(1)
        end = datetime.strptime(self.y.index[-1].strftime('%Y-%m-%d'), '%Y-%m-%d') + timedelta(forecast_range)
        
        if verbose:
            print(f'Forecast range: {forecast_range} days')
            print(f"Start: {start.strftime('%Y-%m-%d')}\nEnd: {end.strftime('%Y-%m-%d')}")
            
        # model name section
        if self.model == 'TREND_SEASONAL':
            model_name = (f'TRSE({self.p},{self.d},{self.q})' if self.method == 'stl' else f'MSTL with ARIMA({self.p},{self.d},{self.q})')
        if self.model == 'ARIMA':
            model_name = f'ARIMA({self.p},{self.d},{self.q})'
        if self.model == 'STL_with_ARIMA':
            model_name = f'STL with ARIMA({self.p},{self.d},{self.q})'
            
        try:
            if self.model == 'ARIMA':
                model = ARIMA(self.y[-self.max_train_size:], order=(self.p, self.d, self.q))
                fitted = model.fit()
                forecast = fitted.get_forecast(steps=forecast_range)
                yhat = forecast.predicted_mean
                low = forecast.conf_int(alpha = self.alpha).iloc[:, 0]
                high = forecast.conf_int(alpha = self.alpha).iloc[:, 1]
                low.index = high.index = pd.date_range(start, end)
                
            if self.model == 'STL_with_ARIMA':
                model = STLForecast(self.y[-self.max_train_size:], ARIMA, model_kwargs={'order': (self.p, self.d, self.q)}, period = 7)
                fitted = model.fit()
                yhat = fitted.forecast(steps=forecast_range)
                
            if self.model == 'TREND_SEASONAL':
                # 1. Forecasting 'trend' component
                #model_tr = STLForecast(self.trend[train_index], ARIMA, model_kwargs={'order': (p, self.d, q)}, period = self.period)
                model_tr = ARIMA(self.trend[-self.max_train_size:], order = (self.p, self.d, self.q))
                fitted_tr = model_tr.fit()
                forecast_tr = fitted_tr.get_forecast(steps=forecast_range)
                yhat_tr = forecast_tr.predicted_mean
                
                low = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 0]
                high = forecast_tr.conf_int(alpha = self.alpha).iloc[:, 1]
                low.index = high.index = pd.date_range(start, end)
                
                # 2. Forecasting 'seasonal' component
                if self.method == 'stl':
                    model_seas = STLForecast(self.seas[-self.max_train_size:], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period)
                    fitted_seas = model_seas.fit()
                    yhat_seas = fitted_seas.forecast(steps=forecast_range)
                    
                if self.method == 'mstl':
                    model_seas1 = STLForecast(self.seas1[-self.max_train_size:], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period[0])
                    fitted_seas1 = model_seas1.fit()
                    yhat_seas1 = fitted_seas1.forecast(steps=forecast_range)
                    
                    model_seas2 = STLForecast(self.seas2[-self.max_train_size:], ARIMA, model_kwargs={'order': (self.p, 0, self.q)}, period = self.period[1])
                    fitted_seas2 = model_seas2.fit()
                    yhat_seas2 = fitted_seas2.forecast(steps=forecast_range)
                    
                    yhat_seas = yhat_seas1 + yhat_seas2
                    
                # 3. Combining the two
                yhat = yhat_seas + yhat_tr
            
            ret = (np.mean(yhat) - self.y[-1])/self.y[-1]
            decision = ('Wait' if ret <= 0.01 and ret >= -0.01 else ('Buy' if ret > 0.01 else 'Sell'))
            if verbose:
                print(f'Decision: {decision}\nRate: {ret*100:.2f}%')
            
            yhat.index = pd.date_range(start, end)
            
            try:
                fig, ax = plot_forecast_univariate(y_train = self.y[-self.max_train_size:],
                                        y_hat = yhat,
                                        low = low,
                                        high = high,
                                        model_name = model_name,
                                        tick = self.tick)
                fig.text(x = 0.002, y = 0.93, 
                         s=f'Decision: {decision}\nRate: {ret*100:.2f}%\nTime: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                         bbox=dict(facecolor='none', edgecolor='black'),
                         fontsize=8.5)
                return fig, ax, yhat, low, high, decision, ret
            except:
                fig, ax = plot_forecast_univariate(y_train = self.y[-self.max_train_size:],
                                        y_hat = yhat,
                                        model_name = model_name,
                                        tick = self.tick)
                fig.text(x = 0.002, y = 0.93, 
                         s=f'Decision: {decision}\nRate: {ret*100:.2f}%\nTime: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
                         bbox=dict(facecolor='none', edgecolor='black'),
                         fontsize=8.5)
                return fig, ax, yhat, decision, ret
                        
        except np.linalg.LinAlgError as e:
            print('...LU decomposition error')
            return -1
##########################################################################

##########################################################################
class CollectiveInstrumentAnalysis:
    def  __init__(self, 
                  model: str, 
                  ticks: list[str], 
                  start: str, 
                  end: str, 
                  freq: str = '1D',
                  method: str | None = None,
                  period: int = 7):
        
        self.model = model
        self.ticks = ticks      
        self.start = start
        self.end = end
        self.freq = freq
        
        self.method = method
        self.period = period
        
        self.results_dict = {}
        
    def predict(self, 
                max_p: int = 4, 
                max_q: int = 4,
                max_train_size: int = 90,
                test_size: int = 7,
                criterion: str = 'pred',
                save: bool = True,
                folder_name: str = 'all'):
        TIC = time()
        for i, tick in enumerate(self.ticks):
            y = getXTB([tick], self.start, self.end, self.freq).iloc[:, 0]
            print(f'Instrument {i+1} out of {len(self.ticks)}')
            analysis = Univariate(model=self.model,
                                  y=y,
                                  max_train_size=max_train_size,
                                  test_size=test_size,
                                  method=self.method,
                                  period=self.period,
                                  tick=tick)
            analysis.trainCV(max_p=max_p, 
                             max_q=max_q, 
                             criterion=criterion, 
                             verbose=False)
            stats = analysis.CV_winner(plot=None,
                                        return_stats=True,
                                        verbose=True)
            try:
                fig, ax, yhat, low, high, decision, ret = analysis.predict(verbose=True)
            except:
                fig, ax, yhat, decision, ret = analysis.predict(verbose=True)

            if save:
                fig.savefig(f"forecast_plots/" + f"{folder_name}" + f"/{tick}_{pd.Timestamp.today().strftime('%Y_%m_%d')}.png")
                
                text_forecast = open(f"forecast_texts/" + f"{folder_name}" + f"/{tick}_{pd.Timestamp.today().strftime('%Y_%m_%d')}.txt", "w")
                text_forecast.write(f'Decision: {decision}')
                text_forecast.write(f'\nRate: {ret*100:.2f}%')
                text_forecast.write(f'\n\nForecast:\n{yhat}')
                text_forecast.write(f'\n\nConf. int. low:\n{low}')
                text_forecast.write(f'\n\nConf. int. high:\n{high}')
                text_forecast.write('\n\nForecast statistics:')
                text_forecast.write(f'\n\tMean: {stats["mean"]:.3f}')
                text_forecast.write(f'\n\tDirection: {stats["dir"]:.3f}')
                text_forecast.write(f'\n\tFirst day: {stats["first_day"]:.3f}')
                text_forecast.write(f'\n\tRMSE: {stats["rmse"]:.3f}')
                text_forecast.write(f'\n\nTime: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
                text_forecast.close()
            
            self.results_dict[tick] = {'fig': fig,
                                       'ax': ax,
                                       'prediction': yhat,
                                       'decision': decision,
                                       'rate': ret,
                                       'stats': stats,
                                       'start': yhat.index[0],
                                       'end': yhat.index[-1],
                                       'time': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
            TOC = time()
            print(f'...Time from the start: {(TOC-TIC)//3600} hours and {((TOC-TIC)%3600)//60} minutes')