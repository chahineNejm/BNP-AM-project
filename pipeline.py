import pandas as pd
import yfinance as yf
import numpy as np

def covariance_per_month(returns_df):
    
    # Group the data by year and month
    grouped = returns_df.groupby([returns_df.index.year, returns_df.index.month])
    monthly_covariances=[]
    
    for (year, month), group in grouped:
        # Calculate covariance matrix for this group
        cov_matrix = group.cov().values
        monthly_covariances.append(np.array(cov_matrix))

    return monthly_covariances


def pipeline(start_date, end_date, tickers):
    
    data_frames = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        data.name = ticker
        data_frames[ticker] = data
        
    all_data = pd.concat(data_frames.values(), axis=1, keys=data_frames.keys())
    all_data.fillna(method='ffill', inplace=True)

    returns_df = all_data.pct_change().fillna(method='ffill')
    returns_df.fillna(0, inplace=True)
    returns_df.columns = tickers
    returns_df.index = pd.to_datetime(returns_df.index)
    
    monthly_returns_df = returns_df.resample('M').mean()    
    monthly_covariance_matrix = returns_df.groupby([returns_df.index.year, returns_df.index.month]).apply(covariance_per_month).map(lambda x: x[0])
    extract_variance = lambda x : x.var()
    monthly_variance = returns_df.groupby([returns_df.index.year, returns_df.index.month]).apply(extract_variance)
    monthly_returns_df.index= pd.MultiIndex.from_arrays([monthly_returns_df.index.year, monthly_returns_df.index.month], names=['Year', 'Month'])
    return monthly_returns_df, monthly_variance, monthly_covariance_matrix

# Application aux tickers qu'Hugo a proposé (il y en a 146 d'après ce que j'ai vu en faisant tourner)

if __name__=="__main__":
    START_DATE = '2015-01-01' 
    END_DATE = '2017-12-30'
    tickers = ["MSFT", "AAPL"]
    _,_,a=pipeline(START_DATE, END_DATE, tickers)
    print(a)
