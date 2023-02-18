import os
from asyncio.log import logger
import logging
import numpy as np
from tqdm import tqdm
from .collections import *
from .assets import *
from .utils import StrategyParams , Config 
from typing import List
import traceback

from mc import utils



def check_is_below_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) < (1 - threshold)

def check_is_above_threshold(current_price:float,prev_price:float,threshold:float):
    return  (current_price / prev_price) > (1 + threshold)

def check_time_period_frequency(i,period):
    return i % period == 0
class Trader:
    def __init__(self, portfolio: Portfolio):
        self._portfolio = portfolio


    @property
    def portfolio(self):
        return self._portfolio

    def add_cash(self,amount):
        assert amount>=0. , 'Cash must be non-negative'
        self._portfolio.cash.amount+= amount

    def buy_equity(self, asset: Asset,amount:float,transaction_price:float=None) -> None:
        '''
        This function is an adjustment transaction for an asset up
        '''
        transaction_price = asset.current_price if transaction_price is None else transaction_price

        cost = amount * transaction_price

        if cost > self.portfolio.cash.value: raise NotEnoughMoney()

        new_cost_average_price = weighted_avg(x1= asset.initial_price,x2=transaction_price,w1=asset.amount,w2=amount
        )
        #withdraw cash
        self._portfolio.cash.amount -= cost

        #add equity
        asset.initial_price = new_cost_average_price
        asset.amount += amount

    def sell_equity(self, asset: Asset,amount:float,transaction_price:float = None) -> None:
        transaction_price = asset.current_price if transaction_price is None else transaction_price

        if amount > self._portfolio.equity.get_asset(asset.ticker).amount: raise NotEnoughAmount()
        
        asset.amount -= amount
        self._portfolio.cash.amount += amount * transaction_price



    def short_sell(self, asset: Asset,amout:float,transaction_price:float=None) -> None:
        transaction_price = asset.current_price if transaction_price is None else transaction_price

        if self._portfolio.cash < amout * transaction_price: raise NotEnoughMoney()
        asset.amount -= asset.amount
        self._portfolio._cash.amount += asset.amount * transaction_price
        
    def execute_trade(self, asset: Asset, amount: float, transaction_type: TransactionType,transaction_price:float=None) -> None:
        if transaction_type == TransactionType.BUY:
            self.buy_equity(asset, amount,transaction_price)
        elif transaction_type == TransactionType.SELL:
            self.sell_equity(asset, amount,transaction_price)

        elif transaction_type == TransactionType.SHORT_SELL:
            self.short_sell(asset, amount,transaction_price)
        else:
            raise ValueError("Invalid transaction type: {}".format(transaction_type))

    @property
    def portfolio_state_report(self):
        '''
        Print a quick summary of the portfolio report
        '''
        return f'Value:{self._portfolio.value}; Balance:{self._portfolio.share_balance};'
    def rebalance(self,asset: Asset,target_share:float) -> None:
        '''
        rebalance cash and asset.
        `target_share`: result asset share in the portfolio
        '''
        target_asset_value = self.portfolio.value * target_share

        target_asset_amount =  target_asset_value / asset.current_price
        amout_diff = target_asset_amount - asset.amount
        if amout_diff > 0:
            # buy more equity 
            self.execute_trade(asset, amout_diff,TransactionType.BUY)
            
        else:
            # sell some equity
            self.execute_trade(asset, amout_diff * -1.0,TransactionType.SELL)
            

    def write_straddle(self,symbol:Symbols,pct_from_strike:float,t,price,amount) -> None:

        #update option status and add cash
        call_strike = price * (1+ pct_from_strike)
        premium = 0.0
        premium+= self.portfolio.option_book.write(ticker=symbol
                                        ,type= OptionType.CALL
                                        ,side= TransactionType.SHORT_SELL
                                        ,current_price=price
                                        ,strike=call_strike
                                        ,amount=amount
                                        ,expiration=t)

        put_strike = price * (1-pct_from_strike)
        premium+= self.portfolio.option_book.write(ticker=symbol
                                        ,type= OptionType.PUT
                                        ,side= TransactionType.SHORT_SELL
                                        ,current_price=price
                                        ,strike=put_strike
                                        ,amount=amount
                                        ,expiration=t
                                        )

        return premium
    def option_assigment(self,t,symbol:Symbols, price) -> None:
        '''
        Check of any Options are due on assigment and execute either sell or buy operation
        '''
        options =self.portfolio.option_book.underlying_options(symbol)
        delivery_l = []
        for option in options:
            if option is not None and option.decay(t):
                asset_delivery = option.assign(price)

                asset = self.portfolio.equity.get_asset(symbol)
                if option.type == OptionType.CALL:
                    self.sell_equity(asset, amount= asset_delivery.amount ,transaction_price=asset_delivery.current_price)
                    action = TransactionType.SELL
                    
                elif option.type == OptionType.PUT:
                    self.buy_equity(asset,amount= asset_delivery.amount ,transaction_price=asset_delivery.current_price)
                    action = TransactionType.BUY
                delivery_summary = OptionAssigmentSummary(
                                                        ticker = asset_delivery.ticker
                                                        ,amount =asset_delivery.amount
                                                        ,transaction_price=asset_delivery.current_price
                                                        ,action = action
                                                        )
                delivery_l.append(str(delivery_summary))
        
        return delivery_l
class SimulationTracker:
    def __init__(self,time_series:np.array,traders: List[Trader] ,strategy_params:StrategyParams) -> None:
        
        assert isinstance(time_series,np.ndarray)
        self._ts = time_series

        assert isinstance(traders,list) and len(traders)>0
        self._traders = traders
        n,t =time_series.shape
        self._n = n
        self._t = t

        self._rebalancing_count = np.zeros((n,)) # to keep track of how many times rebalancing has been done
        self._last_rebalanced_price = np.copy(time_series[:,0]) # to keep track of the last rebalanced price

        self._ASSET_INDEX = {'equity':0,'cash':1}
        capital_in_asset = time_series[:,0] * strategy_params.percent_allocated
        capital_in_cash = time_series[:,0] * (1-strategy_params.percent_allocated)

        self.strategy_params = strategy_params
        self._allocated_capital = np.stack(( capital_in_asset[:,np.newaxis]
                                    ,capital_in_cash[:,np.newaxis]
                                    ),axis=2
                                    )

        self._allocated_capital = np.repeat(self._allocated_capital,t,axis=1)

        self.logger = utils.create_logger()
    def _is_new_month(self,i):
        return i % 31 == 0

    def max_rebalances_cheker(self,i):
        return self._rebalancing_count[i] < self.strategy_params.max_rebalances

    def _get_price(self,i:int,j:int):
        return self._ts[i,j]
    
    def _capitalize_cash(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['cash']
        cash = self._traders[i].portfolio.cash
        rate_ = (1+self.strategy_params.cash_interest/365)

        cash.capitalize(rate_)
        self._allocated_capital[i,j,asset_idx] =  cash.amount

        dollars_added = (self._allocated_capital[i,j,asset_idx] - self._allocated_capital[i,j-1,asset_idx])
        
        self.logger.info(f'{j}:cash capitalized: added '+str(dollars_added))
    
    def _change_asset_price(self,i:int,symbol:Symbols, price:float):
        assst = self._traders[i].portfolio.equity.get_asset(symbol)
        self._traders[i].portfolio.equity.log_asset_price(assst,price)

    def _log_equity_value(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['equity']
        self._allocated_capital[i,j,asset_idx]  =self._traders[i].portfolio.equity.value
    
    def _log_cash_value(self,i:int,j:int):
        asset_idx = self._ASSET_INDEX['cash']
        self._allocated_capital[i,j,asset_idx]  = self._traders[i].portfolio.cash.value


    def log_state_change(self,i:int,j:int):
        self._log_cash_value(i,j)
        self._log_equity_value(i,j)
    
    def _rebalance_portfolio(self,i,j,symbol:Symbols,price):
        asset = self._traders[i].portfolio.equity.get_asset(symbol)
        is_below_threshold_triggered = check_is_below_threshold(price,self._last_rebalanced_price[i],self.strategy_params.rebalance_threshold_down) 
        is_above_threshold_triggered = check_is_above_threshold(price,self._last_rebalanced_price[i],self.strategy_params.rebalance_threshold_up)

        capped_threshold_rebalances = (is_below_threshold_triggered ^ is_above_threshold_triggered)  and self.max_rebalances_cheker(i)
        interval_rebalance = False
        if capped_threshold_rebalances or interval_rebalance:
            self.logger.info(f'{j}:Rebalancing criteria satisfied;current balance:'+str(self._traders[i].portfolio.share_balance))


            self._traders[i].rebalance(asset,self.strategy_params.rebalance_asset_ration)
            self._rebalancing_count[i] += 1
            self._last_rebalanced_price[i] = price

            self.logger.info(f'{j}:New balance:'+str(self._traders[i].portfolio.share_balance))
            self.log_state_change(i,j)

    def _validate_derivatives(self,i,t,symbol,price):
        
        interval_passed = check_time_period_frequency(t,self.strategy_params.option_every_itervals)
        #check the assigmnet
        delivery = self._traders[i].option_assigment(t,symbol,price)
        self.logger.info(f'{t}:Option delivery summary:'+ ','.join(delivery))

        #write new options 
        if interval_passed:
            self.logger.info(f'{t}:Eligibal interval for option writing')
            asset = self._traders[i].portfolio.equity.get_asset(symbol)

            amount = asset.amount * self.strategy_params.option_amount_pct_of_notional
            self.logger.info(f'{t}:Eligibal cash amount for option writing:'+str(amount))

            premium_collected = self._traders[i].write_straddle(symbol
                                        ,pct_from_strike = self.strategy_params.option_straddle_pct_from_strike
                                        , t= self.strategy_params.option_duration + t
                                        ,price=price,amount=amount)
        
            self._traders[i].add_cash(premium_collected)
            self.logger.info(f'{t}:Premium collected:'+ str(premium_collected))

            #log portfolio change
            self.log_state_change(i,t)

        #log results
        active_options = self._traders[i].portfolio.option_book.num_active_options
        self.logger.info(f'{t}:Option on the book:'+ str(active_options)+';option book: '+str(self._traders[i].portfolio.option_book))

    def _end_of_day_report(self,i,t):
        self.logger.info(f'{t}:End-Of-Day Report:'+ self._traders[i].portfolio_state_report)

    def run_simulations(self,logs_dir=None):
        '''
        Runner finction that exectute strategy for each price prajectory
        '''

        for i in tqdm(range(self._n)):
            log_file = os.path.join(logs_dir,f'simulation_{i}.log') if logs_dir is not None else None
            self.logger = utils.create_logger(log_file)
            for j in range(1, self._t):
                try:
                    symbol_ = Symbols[self.strategy_params.ticker_name]
                    #get information from market
                    new_price = self._get_price(i,j)
                    self.logger.info(f"{j}:{symbol_.value}:morning price={new_price}")
                    prev_price =self._get_price(i,j-1)
                    # payoff = (new_price/prev_price)
                    
                    #update portfolio state pre action
                    self._change_asset_price(i,symbol_,new_price)


                    self._capitalize_cash(i,j)
                    
                    #assign market return to allocated portfolio
                    self._log_equity_value(i,j)

                    #check derivative contract execution
                    self._validate_derivatives(i,j,symbol_,new_price)

                    #rebalance if needec
                    self._rebalance_portfolio(i,j,symbol_,new_price)

                    self._end_of_day_report(i,j)

                except Exception as e:
                    
                    self.logger.error(f'{j}:Exception during simulation:'+str(e)+'\n'+traceback.format_exc())
                    break
        return self

    @property
    def allocated_capital(self):
        return self._allocated_capital


def asset_return(asset:Asset,price:float):
    '''
    Return the value of the `Asset` in the numeraire 
    '''
    asset.current_value = price
    return asset.pct_return()


def initialize_executors(n,initial_price,strategy_params: StrategyParams) -> List[Trader]:
    '''
    Return a list of portfolios for each simulation
    '''

    sim_portfolios = []
    for i in range(n):
        portfolio = Portfolio()
        cash_ = Cash(amount =  strategy_params.amount_multiple * (1-strategy_params.percent_allocated) * initial_price )
        ticker = Symbols[strategy_params.ticker_name]
        asset = Equity(ticker=ticker,amount= strategy_params.amount_multiple * strategy_params.percent_allocated
                                ,initial_price=initial_price)

        equity_ = EquityPortfolio().add_asset(asset)


        #compose portfolio
        portfolio.cash = cash_
        portfolio.equity = equity_

        portfolio.option_book = OptionBook(strategy_params.option_premium)
        

        trader = Trader(portfolio)

        sim_portfolios.append(trader)

    return sim_portfolios

def run_one_asset_rebalance_portfolio_v1(time_series: np.ndarray, 
                        strategy_params: StrategyParams,
                        config: Config

                        ) -> np.ndarray:

    n, t = time_series.shape
    #initiate portfolios
    sim_portfolios = initialize_executors(n
                                            ,initial_price = config.return_function_params['current_price']
                                            ,strategy_params=strategy_params
                                            )
    
    #walk throught time series 
    sim_tracker = (SimulationTracker(time_series,sim_portfolios,strategy_params)
                        .run_simulations(logs_dir=config.logs_dir)
                        )
    return sim_tracker.allocated_capital

def run_one_asset_rebalance_portfolio_v0(time_series: np.ndarray, 
                        strategy_params: StrategyParams
                        ) -> np.ndarray:
    """
    Rebalances the portfolio so that the proportion of capital allocated to the asset
    is always `percent_allocated` after the price of the asset drops by more than `threshold`
    since the last rebalancing. 
    Rebalancing is done at most k times.
    """
    n, t = time_series.shape


    rebalancing_count = np.zeros((n,)) # to keep track of how many times rebalancing has been done
    last_rebalanced_price = np.copy(time_series[:,0]) # to keep track of the last rebalanced price

    
    capital_in_asset = time_series[:,0] * strategy_params.percent_allocated
    capital_in_cash = time_series[:,0] * (1-strategy_params.percent_allocated)

    allocated_capital = np.stack(( capital_in_asset[:,np.newaxis]
                                ,capital_in_cash[:,np.newaxis]
                                ),axis=2
                                )
    allocated_capital = np.repeat(allocated_capital,t,axis=1)

    print('running portfolio...')
    for i in tqdm(range(n)):
        for j in range(1, t):

            #assign market return to allocated portfolio
            payoff = (time_series[i, j]/time_series[i, j-1]  )
            allocated_capital[i,j,0] = allocated_capital[i,j-1,0] * payoff
            
            #add capitalization on the cash returns 
            allocated_capital[i,j,1] =  allocated_capital[i,j-1,1] * (1+strategy_params.cash_interest/365)

            if ((time_series[i, j] / last_rebalanced_price[i] < 1 - strategy_params.rebalance_threshold) \
                and (rebalancing_count[i] < strategy_params.max_rebalances)) \
                or (j % strategy_params.rebalance_every ==0):
                
                allocated_capital[i,j:,:] = [allocated_capital[i,j].mean()
                                        ,allocated_capital[i,j].mean()]

                rebalancing_count[i] += 1
                last_rebalanced_price[i] = time_series[i, j]
                continue

    return allocated_capital

