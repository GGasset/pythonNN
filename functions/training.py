from classes.enums import activation, cost_functions
import math

def activation_derivative(input_activation_value, activation_function: activation) -> float:
    x = input_activation_value
    if activation_function == activation.sine:
        return -math.cos(x)
    elif activation_function == activation_function.sigmoid:
        return ((x-1)*math.pow(math.e, x-1) - (x-1)*math.pow(math.e, x-1) * math.pow(math.e, x)) / (math.pow(1-math.e, 2))
    elif activation_function == activation.none:
        return 1
    
def cost_function(output: float, label: float, cost_function: cost_functions):
    if cost_function == cost_functions.squared_mean:
        return math.pow(label - output, 2)
    
def cost_function_derivative(output: float, label: float, cost_function: cost_functions):
    if cost_function == cost_functions.squared_mean:
        return -2*(label - output)
