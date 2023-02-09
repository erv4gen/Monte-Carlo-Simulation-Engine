
from .assets import *
from .utils import StrategyParams , Config 
from typing import List, Dict
from collections.abc import Mapping

class AssetCollection:
    def __init__(self) -> None:
        self._assets:  Dict[Mapping[str,Asset]]

    @property
    def tickers(self):
        return list(self._equity.keys())

    def log_asset_price(self,asset: Asset,market_price:float) -> None:
        '''
        Log asset price
        '''
        asset.current_price = market_price


    def add_asset(self,asset:Equity):
        '''
        Add asset to the portfolio
        '''
        if asset.ticker in self.tickers: raise DuplicateTickersNotAllowed()

        self._assets[asset.ticker] = asset

    @property
    def value(self):
        value= 0.0
        for ticker in self._equity.items():
            value += self._equity[ticker].value
        
        return value
class EquityPortfolio(AssetCollection):
    def __init__(self) -> None:
        super().__init__()

        
class OptionBook(AssetCollection):
    def __init__(self,premium_pct) -> None:
        super().__init__()
        self._premium_pct = premium_pct
        self._options = []
        self._expired_options = []

    def write(self, type:OptionType,side: TransactionType, current_price: float, strike: float, amount: float, expiration: int) -> Option:
        
        if type ==  OptionType.PUT:
            option = EuropeanNaivePutOption(premium_pct=self._premium_pct, current_price=current_price, strike=strike,
                        amount=amount, expiration=expiration)
        
        elif type == OptionType.CALL:
            option = EuropeanNaiveCallOption(premium_pct=self._premium_pct, current_price=current_price, strike=strike,
                        amount=amount, expiration=expiration)
        
        self._options.append(option)
        return option

    def update_options(self, t1: int) -> None:
        for option in self._options:
            if option.decay(t1):
                self._expired_options.append(option)
                self._options.remove(option)

    def expired_options(self) -> List[EuropeanNaiveOption]:
        return self._expired_options

    
    def underlying_options(self,asset:Equity):
        '''
        Return options for the given underlying 
        '''
        return [option for option in self._options if option.underlying == asset.ticker]
    
    @property
    def all_options(self) -> List[EuropeanNaiveOption]:
        return self._options


class Portfolio:
    def __init__(self) -> None:
        
        self._equity: EquityPortfolio
        self._cash: Cash
        self._options :OptionBook
        
    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self,value:Cash):
        self._cash = value

    @property
    def value(self):
        return self._cash.value + self._equity.value + self._options.value
    
    
    @property
    def portfolio_balance(self):
        #TODO fix
        asset_share = self.equity.value / (self.equity.value + self.cash.value)
        cash_share = self.cash.value / (self.equity.value + self.cash.value)
        return PortfolioBalance(asset_share,cash_share)