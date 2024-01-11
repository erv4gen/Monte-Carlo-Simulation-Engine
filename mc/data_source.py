import ccxt
import numpy as np
from datetime import datetime, timedelta
from typing import NamedTuple, DefaultDict
from . import names
from collections import defaultdict

class TickerMarketData(NamedTuple):
    current_price:float=100.0
    volatility: float=0.25

def get_crypto_price_series(exchange_id:str,symbol:str, lookback_days=365)-> np.array:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()

    timeframe = '1d'
    now = exchange.milliseconds()
    historical_prices = exchange.fetch_ohlcv(symbol, timeframe, since=now - (86400000 * lookback_days))
    return np.array([price[4] for price in historical_prices])

def get_crypto_price_volatility(exchange_id:str,symbol:str, lookback_days=365)->TickerMarketData:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
        
    close_prices = get_crypto_price_series(exchange_id,symbol,lookback_days)
    
    log_returns = np.log(close_prices[1:]) - np.log(close_prices[:-1])
    annualized_volatility = round(np.std(log_returns) * np.sqrt(365),3)
    current_price = round(exchange.fetch_ticker(symbol)['last'],3)
    return TickerMarketData(current_price, annualized_volatility)


def load_market_data(exchange='coinbasepro',lookback_days=365)->DefaultDict[str,TickerMarketData]:
    market_data = defaultdict(TickerMarketData)
    for symbol in names.Symbols:
        try:
            if symbol == symbol.CASH: continue
            market_data[symbol.value] = get_crypto_price_volatility(exchange
                                                                    ,symbol=f'{symbol.value}/USDT'
                                                                    ,lookback_days=lookback_days)
            print(f'{symbol.value}: Data Loaded')
        except:
            print(f'CANNOT LOAD {symbol.value} DATA')
    return market_data


def load_array_series(series,exchange='coinbasepro',lookback_days=365)->np.array:
    data_list = []
    for symbol in series:
        try:
            if symbol == symbol.CASH: continue
            data_list.append(get_crypto_price_series(exchange
                                                                    ,symbol=f'{symbol.value}/USDT'
                                                                    ,lookback_days=lookback_days)
            )
        except:
            print(f'CANNOT LOAD {symbol.value} DATA')
    return np.stack(data_list, axis=0 )