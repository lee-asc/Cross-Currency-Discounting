import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin


if __name__ == '__main__':

    yfin.pdr_override()

    tickers = ['AAPL', 'MSFT', 'IBM', '^GSPC']
    start = dt.datetime(2015, 12, 1)
    end = dt.datetime(2021, 1, 1)

    data = pdr.get_data_yahoo(tickers, start, end)

    data = data['Adj Close']

    log_returns = np.log(data / data.shift())

    plt.plot(log_returns)
    plt.show()
    #sys.exit()

    cov = log_returns.cov()
    var = log_returns['^GSPC'].var()

    print('cov: ', cov)

    beta = cov.loc['AAPL', '^GSPC'] / var

    print('beta: ', beta)
    #sys.exit()

    risk_free_return = 0.0138
    market_return = .105
    expected_return = risk_free_return + beta * (market_return - risk_free_return)

    print('expected_return: ', expected_return)