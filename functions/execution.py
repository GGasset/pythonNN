from classes.enums import activation
import math


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
