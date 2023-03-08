from enum import Enum

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