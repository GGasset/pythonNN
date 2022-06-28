from enum import Enum


class activation(Enum):
    sigmoid = 'sigmoid'
    sine = 'sine'
    none = 'none'
    
class cost_functions(Enum):
    squared_mean = 'squared_mean'