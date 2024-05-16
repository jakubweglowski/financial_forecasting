import eurostat
from DataCollectorXtb import DataCollectorXtb as dcxtb
import pandas as pd
import numpy as np


# SECTION 1: XTB data

def getXTB(tickers: list, start: str, end: str, freq: str = '1D'):
    """_summary_

    Args:
        tickers (list): list of XTB tickers to be downloaded
        start (str): starting date in 'YYYY-MM-DD' format
        end (str): ending date in 'YYYY-MM-DD' format
        freq (str): time-frequency of the data ('1D' for daily, '1MN' for  monthly etc.)

    Returns:
        data (pd.DataFrame or pd.Series): dataframe od series with the requested data
    """
    data = pd.DataFrame()
    
    if len(tickers) >= 1:
        for tick in tickers:
            try:
                data[tick] = dcxtb(symbol=tick, start=start, end=end, period=freq, cols_to_save=['Close']).data.iloc[:, 0]
            except:
                print("Error: maybe 'freq' parameter is invalid...")        
        return data
    else:
        # len(tickers) <= 0
        raise ValueError(print(f"No valid ticker provided."))
    
    
# SECTION 2: EUROSTAT data

country_names_dict = {'Belgium': 'BE',
                        'Greece': 'EL',
                        'Lithuania': 'LT',
                        'Portugal': 'PT',
                        'Bulgaria': 'BG',
                        'Spain': 'ES',
                        'Luxembourg': 'LU',
                        'Romania': 'RO',
                        'Czechia': 'CZ',
                        'France': 'FR',
                        'Hungary': 'HU',
                        'Slovenia': 'SI',
                        'Denmark': 'DK',
                        'Croatia': 'HR',
                        'Malta': 'MT',
                        'Slovakia': 'SK',
                        'Germany': 'DE',
                        'Italy': 'IT',
                        'Netherlands': 'NL',
                        'Finland': 'FI',
                        'Estonia': 'EE',
                        'Cyprus': 'CY',
                        'Austria': 'AT',
                        'Sweden': 'SE',
                        'Ireland': 'IE',
                        'Latvia': 'LV',
                        'Poland': 'PL',
                        'Euro_Area': 'EA'}

def generategeo(countries: list) -> list:
    """Transform country names to country codes

    Args:
        countries (list): List of country names (in English)

    Returns:
        list: Eurostat country codes
    """
    geo = []
    for c in countries:
        try:
            geo.append(country_names_dict[c])
        except KeyError as e:
            print('Given country either doesn\'t exist or is not an EU member')
    return geo


def getGDP(countries: list, start, end):

    geo = generategeo(countries)   

    raw_data = eurostat.get_data('namq_10_gdp',
                                filter_pars={'s_adj': ['SCA'],
                                             'unit': ['CP_MNAC'], 
                                             'na_item': ['B1GQ'],
                                             'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                        columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y = np.log(y).diff().dropna()
    y.columns = [c + '_gdp' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]

def getUNEMPLOYMENT(countries: list, start, end):
    
    geo = generategeo(countries)   

    raw_data = eurostat.get_data('une_rt_m',
                                filter_pars={'age': ['TOTAL'],
                                             's_adj': ['SA'],
                                             'sex': ['T'],
                                             'unit': ['PC_ACT'],
                                             'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                        columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y = np.log(y).diff().dropna()
    y.columns = [c + '_unempl' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]
    
    
    
def getHICP(countries: list, start: str, end: str) -> pd.DataFrame:
    
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('prc_hicp_manr',
                            filter_pars={'coicop': ['CP00'],
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_hicp' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]


def getRATES(countries: list, start: str, end: str, rate: str = 'DTD') -> pd.DataFrame:
    """Download interest rates

    Args:
        countries (list): _description_
        start (str): _description_
        end (str): _description_
        rate (str, optional): Defaults to 'DTD'. Possible values:
                              'DTD' for day-to-day, 
                              'M1' for 1-month,
                              'M3' for 3-month,
                              'M6' for 6-month,
                              'M12' for 12-month.

    Returns:
        pd.DataFrame: data frame with requested interest rates
    """
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('irt_st_m',
                            filter_pars={'int_rt': ['IRT_'+rate],
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_'+rate+'_rate' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]
    
    
def getINDUSTRY(countries: str, start: str, end: str):
    """This does not work!

    Args:
        countries (str): _description_
        start (str): _description_
        end (str): _description_

    Returns:
        _type_: _description_
    """
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('sts_inpr_m',
                            filter_pars={'s_adj': ['SCA'],
                                         'nace_r2': ['B-D'], 
                                         'unit': ['I21'], 
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_industry' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]


def getSERVICES(countries: str, start: str, end: str):
    """This does not work!

    Args:
        countries (str): _description_
        start (str): _description_
        end (str): _description_

    Returns:
        _type_: _description_
    """
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('sts_sepr_m',
                            filter_pars={'s_adj': ['SCA'], 
                                         'nace_r2': ['H-N_X_K'],
                                         'unit': ['I21'], 
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_services' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]


def getTRADE(countries: str, start: str, end: str):
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('sts_trtu_m',
                            filter_pars={'indic_bt': ['TOVV'],
                                         's_adj': ['SCA'], 
                                         'nace_r2': ['G47'],
                                         'unit': ['I21'], 
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_trade' for c in geo]

    return y[y.index.isin(pd.date_range(start, end))]

def getCONSTRUCTION(countries: str, start: str, end: str) -> pd.DataFrame:

    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('sts_cobp_m',
                            filter_pars={'indic_bt': ['PSQM'], 
                                         's_adj': ['SCA'],
                                         'cpa2_1': ['CPA_F41001_41002'],
                                         'unit': ['I21'],
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_construction' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]
        
        
def getBALANCE(countries: str, start: str, end: str) -> pd.DataFrame:
    geo = generategeo(countries)
    
    raw_data = eurostat.get_data('bop_c6_m',
                            filter_pars={'bop_item': ['CA'], 
                                         'currency': ['MIO_NAC'],
                                         'sector10': ['S1'],
                                         'stk_flow': ['BAL'],
                                         'geo': geo})
    data = pd.DataFrame(raw_data[1:],
                    columns=[x if x != 'geo\\TIME_PERIOD' else 'geo' for x in raw_data[0]]).dropna(axis=1)
    
    y = data.select_dtypes(include=np.number).transpose()
    y.index = pd.to_datetime(y.index)
    y.columns = [c + '_balance' for c in geo]
    
    return y[y.index.isin(pd.date_range(start, end))]