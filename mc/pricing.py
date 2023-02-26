import QuantLib as ql
from .assets import *

class QlEuropeanOption:
    def __init__(self,spot_price:float,strike:float, maturity:int, volatility:float,risk_free_rate:float, dividend_rate:float, option_type:OptionType) -> None:
        option_type_ql = ql.Option.Call if option_type == OptionType.CALL else ql.Option.Put
        # Get the current date
        init_date = ql.Date.todaysDate()
        self.eval_date = init_date+ ql.Period(0, ql.Days)
        # Set the evaluation date to eval_date
        ql.Settings.instance().evaluationDate = self.eval_date
        
        # Set up the option parameters
        payoff = ql.PlainVanillaPayoff(option_type_ql, strike)
        exercise = ql.EuropeanExercise(self.eval_date + ql.Period(maturity, ql.Days))
        option = ql.VanillaOption(payoff, exercise)
        
        # Set up the market data
        spot_price = ql.SimpleQuote(spot_price) 
        quote_handle = ql.QuoteHandle(spot_price)
        risk_free_rate = ql.YieldTermStructureHandle(
            ql.FlatForward(self.eval_date, ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual365Fixed()))
        dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(self.eval_date, ql.QuoteHandle(ql.SimpleQuote(dividend_rate)), ql.Actual365Fixed()))
        volatility = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.eval_date, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(volatility)), ql.Actual365Fixed()))
        process = ql.BlackScholesMertonProcess(quote_handle, dividend_yield, risk_free_rate, volatility)
        
        # Calculate the option price
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        self._init_date = init_date
        self._option:ql.VanillaOption = option
        self._init_date: ql.Date = init_date
        self._spot_price: ql.SimpleQuote = spot_price
        self._quote_handle = ql.QuoteHandle(self._spot_price)
        self._process: ql.BlackScholesMertonProcess = process
        self._volatility: float = volatility
        self._risk_free_rate = risk_free_rate
        self._dividend_yield = dividend_yield

    def decay(self,i):
        self._eval_date = self._init_date + ql.Period(i, ql.Days)
        ql.Settings.instance().evaluationDate = self._eval_date
        self._process = ql.BlackScholesMertonProcess(self._quote_handle, self._dividend_yield, self._risk_free_rate, self._volatility)
        engine = ql.AnalyticEuropeanEngine(self._process)
        
        self._option.setPricingEngine(engine)

        return self

    def price_drift(self,new_price):
        self._spot_price.setValue(new_price)
        return self

    def NPV(self):
        return self._option.NPV()