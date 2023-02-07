
from typing import NamedTuple


class PortfolioBalance(NamedTuple):
    equity: float
    cash: float

def weighted_avg(x1,x2,w1,w2):
    '''
    calculate weighted average of two values
    '''
    return (x1 * w1 +x2 * w2) / (w1+ w2)


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
    def __init__(self,amount:float=0.0,initial_price:float=1.0) -> None:
        self._amount = amount
        #initial price
        self._s0_price = initial_price
        #current price (last recorded)
        self._st_price = initial_price
    

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self,value:float):
        assert value >0., NegativePriceExepton()
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

class Option(Asset):
    def __init__(self,premium_pct:float,*args, **kwargs) -> None:
        '''
        `premium_pct` what is the premium as a pct of the price
        '''
        super().__init__(*args,**kwargs)
        self._premium_pct = premium_pct
        #mark if the option is active
        self._ALIVE = False
        self._strike = None
    def write(self,current_price:float,strike:float,amount:float):
        '''
        When option is written, it becomes alive until assigmen
        the function returns `premium_value`
        '''
        if self._ALIVE: return None
        premium_value = current_price * self._premium_pct
        self._strike = strike
        self.amount = amount
        self._ALIVE = True
        return premium_value

    def assign(self):
        '''
        Assigment makes option not alive and return the delivery asset
        '''
        if not self._ALIVE: return None

        assets_to_delivery = self.amount * self._strike
        self._ALIVE = False
        return assets_to_delivery


class CallOption(Option):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
    
    def assign(self,current_price:float):
        assets_to_delivery = super().assign()
        #if option in the monay
        if current_price>= self._strike:
            return assets_to_delivery

class PutOption(Option):
    def __init__(self, premium_pct: float) -> None:
        super().__init__(premium_pct)
    
    def assign(self,current_price:float):
        assets_to_delivery = super().assign()
        #if option in the monay
        if current_price <= self._strike:
            return assets_to_delivery




class Portfolio:
    def __init__(self) -> None:
        self._cash: Cash = None
        self._equity: Equity = None
        self._option: Option = None
    
    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self,value:Cash):
        self._cash = value

    @property
    def equity(self):
        return self._equity

    @equity.setter
    def equity(self,value:Equity):
        self._equity = value


    @property
    def option(self):
        return self._option

    @option.setter
    def option(self,value:Option):
        self._option = value

    @property
    def capital(self):
        equity_value = self.equity.value
        cash_value = self.cash.value
        return equity_value + cash_value

    def log_asset_price(self,market_price) -> None:
        '''
        Log equity price
        '''
        self.equity.current_price = market_price

    def buy_equity(self,adj_amount:float) -> None:
        '''
        This function is an adjustment transaction for an asset up
        '''
        cost = adj_amount * self.equity.current_price
        if cost > self.cash.amount: raise NotEnoughMoney()

        new_cost_average_price = weighted_avg(x1= self.equity.initial_price,x2=self.equity.current_price,w1=self.equity.amount,w2=adj_amount
        )
        #withdraw cash
        self.cash.amount -= cost

        #add equity
        self.equity.initial_price = new_cost_average_price
        self.equity.amount += adj_amount

    def sell_equity(self, amount: float) -> None:
        if amount > self._equity.amount: raise NotEnoughMoney()
        
        self._equity.amount -= amount
        self._cash.amount += amount * self.equity.current_price

    @property
    def portfolio_balance(self):
        asset_share = self.equity.value / (self.equity.value + self.cash.value)
        cash_share = self.cash.value / (self.equity.value + self.cash.value)
        return PortfolioBalance(asset_share,cash_share)

    def rebalance(self,target_share:float) -> None:
        '''
        rebalance cash and asset.
        `target_share`: result asset share in the portfolio
        '''
        

        target_asset_value = self.capital * target_share

        target_asset_amount =  target_asset_value / self.equity.current_price
        amout_diff = target_asset_amount - self.equity.amount
        if amout_diff > 0:
            # sell some equity
            self.buy_equity(amout_diff)
            
        else:
            # buy more equity
            self.sell_equity(amout_diff * -1)