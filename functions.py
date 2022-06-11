import random as r
from classes import activation
import math

def generate_weight(min_weight=-1.5, max_weight=1.5, closest_to_0=0.37):
    """min_weight must be negative and max_weight negative for a correct functioning"""
    is_below_threshold = True
    closest_to_0 = abs(closest_to_0)
    while is_below_threshold:
        negative = r.randint(0, 1)
        weight = r.random() * (min_weight * negative + max_weight * (negative == 0))
        is_below_threshold = weight < closest_to_0
        
    weight *= (-1 * negative) + (1 * negative == 0)
    return weight

def linear_function(previous_activations: list[float], weights: list[float], bias: float):
    output = bias
    for i in range(len(previous_activations)):
        output += previous_activations[i] * weights[i]
    return output

def activate(value: float, activation_function: activation)-> float:
    if activation_function == activation.none:
        return value
    elif activation_function == activation.sine:
        return math.sin(value)
    elif activation_function == activation.sigmoid:
        return 1 / (1 + math.pow(math.e, -value))
        
def derivative(input_activation_value, activation_function: activation) -> float:
    x = input_activation_value
    if activation_function == activation.sine:
        return -math.cos(x)
    elif activation_function == activation_function.sigmoid:
        return ((x-1)*math.pow(math.e, x-1) - (x-1)*math.pow(math.e, x-1) * math.pow(math.e, x)) / (math.pow(1-math.e, 2))
    elif activation_function == activation.none:
        return 1
    
def will_mutate(probability: float):
    return r.random() > probability

def get_variation(max_variation):
    negative = r.randint(0, 1)
    negative -= 1 * (negative == 0)
    return r.random() * max_variation * negative