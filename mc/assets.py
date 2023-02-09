
from typing import NamedTuple
from enum import Enum

from matplotlib import ticker


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

def weighted_avg(x1,x2,w1,w2):
    '''
    calculate weighted average of two values
    '''
    return (x1 * w1 +x2 * w2) / (w1+ w2)


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
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return "Not enough money for transaction"

class NegativePriceExepton(Exception):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

    def __str__(self) -> str:
        return 'Negative price setter is not allowed'

class Asset:
    def __init__(self,ticker,amount:float=0.0,initial_price:float=1.0) -> None:
        self._amount = amount
        #initial price
        self._s0_price = initial_price
        #current price (last recorded)
        self._st_price = initial_price

        self._ticker = ticker
    
    @property
    def ticker(self):
        return ticker

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

class Cash(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

class Equity(Asset):
    def __init__(self, *args: object, **kwargs:object) -> None:
        super().__init__(*args,**kwargs)

class EuropeanNaiveOption(Asset):
    def __init__(self,premium_pct:float,*args, **kwargs) -> None:
        '''
        `premium_pct` what is the premium as a pct of the price
        '''
        super().__init__(*args,**kwargs)
        self._type = None
        self._premium_pct = premium_pct
        #mark if the option is active
        self._ALIVE = False
        self._strike = None
        self._T = None
    
    @property
    def type(self):
        return self._type

    def write(self,underlying:str,current_price:float,strike:float,amount:float,expiration:int):
        '''
        When option is written, it becomes alive until assigmen
        the function returns `premium_value`
        '''
        if self._ALIVE: return 0.0
        self._underlying = underlying
        premium_value = current_price * self._premium_pct
        self._strike = strike
        self.amount = amount
        self._T = expiration
        self._ALIVE = True
        
        return premium_value

    @property
    def underlying(self):
        return self._underlying
        
    def decay(self,t1):
        '''
        Log time dacay
        '''
        return self._ALIVE and t1 >= self._T

    def assign(self) ->Equity:
        '''
        Assigment makes option not alive and return the delivery asset
        '''
        asset_delivery = Equity(amount=self.amount,initial_price=self._strike) if self._ALIVE else Equity()
        self._ALIVE = False
        return asset_delivery




class EuropeanNaiveCallOption(EuropeanNaiveOption):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
        self._type = OptionType.CALL
    def assign(self,current_price:float) -> Equity:
        asset_delivery = super().assign()
        #if option in the monay
        if current_price< self._strike:
            asset_delivery.amount = 0.0

        #max(0, current_price - self.strike)
        return asset_delivery 

class EuropeanNaivePutOption(EuropeanNaiveOption):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
        self._type = OptionType.PUT

    def assign(self,current_price:float) ->Equity:
        asset_delivery = super().assign()
        #if option in the monay
        if current_price > self._strike:
            asset_delivery.amount = 0.0
        return asset_delivery


