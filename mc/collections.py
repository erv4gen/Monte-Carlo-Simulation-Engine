
from .assets import *
from .utils import StrategyParams , Config 
from typing import List, Dict
from collections.abc import Mapping

class AssetCollection:
    def __init__(self) -> None:
        self._assets = dict()

    @property
    def tickers(self):
        return list(self._assets.keys())

    def log_asset_price(self,asset: Asset,market_price:float) -> None:
        '''
        Log asset price
        '''
        asset.current_price = market_price

    def get_asset(self,ticker:Symbols) -> Equity:
        '''
        return asset from the portfolio if exists
        '''
        if ticker not in self.tickers: raise AssetIsNotInPortfolio()
        return self._assets[ticker]

    def add_asset(self,asset:Equity):
        '''
        Add asset to the portfolio
        '''
        if asset.ticker in self.tickers: raise DuplicateTickersNotAllowed()

        self._assets[asset.ticker] = asset
        return self
    @property
    def value(self):
        value= 0.0
        for _, asset in self._assets.items():
            value += asset.value
        
        return value


    def __repr__(self) -> str:
        return 'AssetCollection'
class EquityPortfolio(AssetCollection):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        super().__repr__()
        return "EquityPortfolio("+ ','.join(self.tickers) + ')'
        
class OptionBook(AssetCollection):
    def __init__(self,premium_pct) -> None:
        super().__init__()
        self._premium_pct = premium_pct
        self._options = []
        self._expired_options = []

    def write(self,ticker:Symbols, type:OptionType,side: TransactionType, current_price: float, strike: float, amount: float, expiration: int) -> EuropeanNaiveOption:
        
        if type ==  OptionType.PUT:
            option = EuropeanNaivePutOption(ticker=ticker,premium_pct=self._premium_pct).write(current_price=current_price, strike=strike,
                        amount=amount, expiration=expiration)
        
        elif type == OptionType.CALL:
            option = EuropeanNaiveCallOption(ticker=ticker,premium_pct=self._premium_pct).write(current_price=current_price, strike=strike,
                        amount=amount, expiration=expiration)
        
        self._options.append(option)
        return option.premium

    def update_options(self, t1: int) -> None:
        for option in self._options:
            if option.decay(t1):
                self._expired_options.append(option)
                self._options.remove(option)

    def expired_options(self) -> List[EuropeanNaiveOption]:
        return self._expired_options

    
    def underlying_options(self,ticker:Symbols) -> List[EuropeanNaiveOption]:
        '''
        Return options for the given underlying 
        '''
        return [option for option in self._options if option.underlying == ticker]
    
    @property
    def all_options(self) -> List[EuropeanNaiveOption]:
        return self._options

    @property
    def num_active_options(self) -> int:
        return len(self._options)

    def __repr__(self) -> str:
        opt_string = ','.join([str(o) for o in self._options])
        return f"OptionBook(premium={self._premium_pct},#options={self.num_active_options},book={opt_string})"

class Portfolio:
    def __init__(self) -> None:
        
        self._equity: EquityPortfolio
        self._cash: Cash
        self._options :OptionBook


    @property
    def option_book(self):
        return self._options
    @property
    def equity(self):
        return self._equity

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self,value:Cash):
        self._cash = value

    @equity.setter
    def equity(self,value:EquityPortfolio):
        self._equity = value

    @option_book.setter
    def option_book(self,value:OptionBook):
        self._options = value

    @property
    def value(self):
        return self._cash.value + self._equity.value + self._options.value
    
    
    @property
    def share_balance(self):
        #TODO fix
        asset_share = self.equity.value / (self.equity.value + self.cash.value)
        cash_share = self.cash.value / (self.equity.value + self.cash.value)
        return PortfolioBalance(asset_share,cash_share)