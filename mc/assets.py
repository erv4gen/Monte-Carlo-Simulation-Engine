
from http.client import UnimplementedFileMode
from typing import NamedTuple
from enum import Enum

from matplotlib import ticker


class Symbols(Enum):
    CASH = 'Cash'
    ETH = 'ETH'

class OptionType(Enum):
    CALL = 'Call'
    PUT = 'Put'


class TransactionType(Enum):
    BUY = 'Buy'
    SELL = 'Sell'
    SHORT_SELL = 'Short_Sell'
class PortfolioBalance(NamedTuple):
    equity: float
    cash: float



class OptionAssigmentSummary(NamedTuple):
    ticker: Symbols
    amount: float
    transaction_price: float
    action: TransactionType

    def __str__(self) -> str:
        return f'OptionAssigmentSummary(ticker={self.ticker},amount={self.amount},price={self.transaction_price},side={self.action})'
def weighted_avg(x1,x2,w1,w2):
    '''
    calculate weighted average of two values
    '''
    return (x1 * w1 +x2 * w2) / (w1+ w2)


class AssetIsNotInPortfolio(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    def __str__(self) -> str:
        return "Asset is not in the portfolio"
class DuplicateTickersNotAllowed(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    def __str__(self) -> str:
        return "Cannot add duplicated ticker to the portfolio"
class NotEnoughAmount(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return "Not enough asset amount for transaction"

class NotEnoughMoney(Exception):
    def __init__(self, message="Not enough money for transaction", *args: object) -> None:
        super().__init__(message,*args)


class NegativePriceExepton(Exception):
    def __init__(self,message='Negative price setter is not allowed', *args: object, **kwargs:object) -> None:
        super().__init__(message,*args,**kwargs)


class Asset:
    def __init__(self,ticker:Symbols,amount:float=0.0,initial_price:float=1.0) -> None:
        self._amount = amount
        #initial price
        self._s0_price = initial_price
        #current price (last recorded)
        self._st_price = initial_price

        self._ticker = ticker

    def capitalize(self,rate):
        assert rate>0, 'Rate cannot be non-positive'
        self._amount*= rate
    
    @property
    def ticker(self):
        return self._ticker

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self,value:float):
        assert value >=0., NegativePriceExepton()
        self._amount = value

    @property
    def initial_price(self):
        return self._s0_price

    @initial_price.setter
    def initial_price(self,value):
        self._s0_price = value

    @property
    def initial_value(self):
        value_ = self._amount * self._s0_price
        return value_

    @property
    def value(self):
        value_ = self._amount * self._st_price
        return value_

    @property
    def current_price(self):
        return self._st_price
    
    @current_price.setter
    def current_price(self,current_price):
        self._st_price = current_price        

    def pct_return(self):
        return 0.0 if self._s0_price==0.0 else self._s1_price/ self._s0_price -1.

    def __repr__(self) -> str:
        return f"{self.ticker.value}(amt={self.amount},s0={self._s0_price},st={self._st_price})"

class Cash(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(ticker=Symbols.CASH,*args,**kwargs)
    

    

class Equity(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

class EuropeanNaiveOption(Asset):
    def __init__(self,premium_pct:float,*args, **kwargs) -> None:
        '''
        `premium_pct` what is the premium as a pct of the price
        '''
        super().__init__(*args,**kwargs)
        self._type: OptionType = None
        self._premium_pct = premium_pct
        #mark if the option is active
        self._ALIVE = False
        self._strike = None
        self._T = None
        self._premium_value = 0.0

    @property
    def premium(self):
        return self._premium_value

    @property
    def type(self):
        return self._type

    def write(self,current_price:float,strike:float,amount:float,expiration:int):
        '''
        When option is written, it becomes alive until assigmen
        the function returns `premium_value`
        '''
        if self._ALIVE: return 0.0
        
        self._strike = strike
        self.amount = amount
        self._T = expiration
        self._ALIVE = True
        self._premium_value = current_price * self._premium_pct
        return self

    @property
    def underlying(self):
        return self._ticker
        
    def decay(self,t1):
        '''
        Log time dacay
        '''
        return self._ALIVE and t1 >= self._T

    def ITM(self,current_price:float) -> bool:
        raise NotImplementedError()
    def assign(self) ->Equity:
        '''
        Assigment makes option not alive and return the delivery asset
        '''
        asset_delivery = Equity(ticker=self.underlying, amount=self.amount,initial_price=self._strike) if self._ALIVE else Equity(ticker=self.underlying)
        self._ALIVE = False
        self._premium_value = 0.0

        return asset_delivery

    def __repr__(self) -> str:
        return f'{self.ticker.value}{self._type.value}Option(k={self._strike},T1={self._T},amt={self._amount})'
    def __str__(self) -> str:
        return f'{self.ticker.value}{self._type.value}Option(k={self._strike},T1={self._T},amt={self._amount})'


class EuropeanNaiveCallOption(EuropeanNaiveOption):
    def __init__(self, premium_pct: float,*args,**kwargs) -> None:
        super().__init__(premium_pct,*args,**kwargs)
        self._type = OptionType.CALL
    def assign(self,current_price:float) -> Equity:
        asset_delivery = super().assign()
        #if option in the monay
        if current_price< self._strike:
            asset_delivery.amount = 0.0

        #max(0, current_price - self.strike)
        return asset_delivery 

    def ITM(self,current_price:float) -> bool:
        return self._strike <= current_price
class EuropeanNaivePutOption(EuropeanNaiveOption):
    def __init__(self, premium_pct: float,*args,**kwargs) -> None:
        super().__init__(premium_pct,*args,**kwargs)
        self._type = OptionType.PUT

    def assign(self,current_price:float) ->Equity:
        asset_delivery = super().assign()
        #if option in the monay
        if current_price > self._strike:
            asset_delivery.amount = 0.0
        return asset_delivery

    def ITM(self,current_price:float) -> bool:
        return self._strike >= current_price

class Future(Asset):
    pass
class AMM(Asset):
    pass