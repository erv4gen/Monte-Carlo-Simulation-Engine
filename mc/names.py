from enum import Enum

class Symbols(Enum):
    CASH = 'Cash'
    ETH = 'ETH'
    DOT = 'DOT'
    BTC = 'BTC'

class OptionType(Enum):
    CALL = 'Call'
    PUT = 'Put'


class TransactionType(Enum):
    BUY = 'Buy'
    SELL = 'Sell'
    SHORT_SELL = 'Short_Sell'


def market_symbols():
    return [v.value for v in Symbols if v!=Symbols.CASH]