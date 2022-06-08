from cmath import sin
import math
import random as r
from enum import Enum

def main():
    option = 1
    nn = neuronal_network()
    if option == 0:
        nn.generate_from_shape([728, 500, 100, 40, 1])
        nn.from_str(str(nn))
        nn.save_to_file('./network.txt')
    else:
        nn.read_from_file('./network.txt')
        print(nn.activation)

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

class activation(Enum):
    sigmoid = 'sigmoid'
    sine = 'sine'
    none = 'none'
            
def activate(self, value: float, activation_function: activation):
    if activation_function == activation.none:
        return value
    elif activation_function == activation.sine:
        return sin(value)
    elif activation_function == activation.sigmoid:
        return 1 / (1 + math.pow(math.e, -value))
        
def derivative(self, input_activation_value, activation_function: activation) -> float:
    x = input_activation_value
    if activation_function == activation.sine:
        return -math.cos(x)
    elif activation_function == activation_function.sigmoid:
        return ((x-1)*math.pow(math.e, x-1) - (x-1)*math.pow(math.e, x-1) * math.pow(math.e, x)) / (math.pow(1-math.e, 2))
    elif activation_function == activation.none:
        return 1
        
class neuron_values:
    def __init__(self) -> None:
        pass


class neuronal_network:
    def __init__(self, mutation_max_variation=0.17, mutation_chance=0.27):
        self.mutation_max_variation = mutation_max_variation
        self.mutation_chance = mutation_chance
    
    def execute_network(self, input: list[float], return_all_values):
        pass
    
    def __str__(self) -> str:
        activation = str(self.activation).removeprefix('activation.')
        output_activation = str(self.output_activation).removeprefix('activation.')
        output = f'activation: {activation}\noutput activation: {output_activation}\n'
        
        output += f'mutation max variation: {self.mutation_max_variation}:\n' + f'mutation chance: {self.mutation_chance}\n'
        
        output += 'biases:\n---\n'
        for i, layer_biases in enumerate(self.biases):
            for j, bias in enumerate(layer_biases):
                output += f'{bias},'
            output = output[:-1]
            output += '\n'
        output += '---\n'

        output += 'weights:\n___\n'
        for i, layer_weights in enumerate(self.weights):
            for j, neuron_weights in enumerate(layer_weights):
                for k, weight in enumerate(neuron_weights):
                    output += f'{weight},'
                output = output[:-1]
                output += '/'
            output = output[:-1]
            output += '\n'
        
        return output
    
    
    def from_str(self, string : str):
        lines = string.splitlines(False)
        self.activation = activation(lines[0].removeprefix('activation: '))
        self.output_activation = activation(lines[1].removeprefix('output activation: '))
        self.mutation_max_variation = float(lines[2].removeprefix('mutation max variation: '))
        self.mutation_chance = float(lines[3].removeprefix('mutation chance: '))
        
        biases_str = string.split('---')[1:-1][0]
        network_biases = biases_str.split('\n')
        biases = []
        self.shape = []
        for i, layer in enumerate(network_biases):
            layer_biases_str = layer.split(',')
            self.shape.append(len(layer_biases_str))
            layer_biases = []
            for j, bias in enumerate(layer_biases_str):
                if bias != '':
                    layer_biases.append(float(bias))
            biases.append(biases)
        
        weights_str = string.split('___')[1:][0]
        weights_str = weights_str.split('\n')
        weights = []
        for i, layer in enumerate(weights_str):
            layer_weights_str = layer.split('/')
            layer_weights = []
            for j, neuron_weights_str in enumerate(layer_weights_str):
                neuron_weights = []
                splitted_neuron_weights_str = neuron_weights_str.split(',')
                for k, weight in enumerate(splitted_neuron_weights_str):
                    if weight != '':
                        neuron_weights.append(float(weight))
                layer_weights.append(neuron_weights)
            weights.append(layer_weights)
        
        self.weights = weights
    
    def save_to_file(self, relative_path : str) -> None:
        if not relative_path.__contains__('.txt'):
            relative_path += '.txt'
        with open(relative_path, 'w') as f:
            f.write(self.__str__())
            
    def read_from_file(self, relative_path : str) -> None:
        if not relative_path.__contains__('.txt'):
            relative_path += '.txt'
        with open(relative_path, 'r') as f:
            lines = f.readlines()
            output = ''
            for i, line in enumerate(lines):
                output += line
        self.from_str(output)
            
    def generate_from_shape(self, shape : list, activation=activation.sine, output_activation=activation.sine, bias=1, min_weight=-1.5, max_weight=1.5, closest_to_0=0.37) -> None:
        """shape: list containing the number of neurons at each layer, including input network. min_weight must be negative and max_weight negative for a correct functioning"""
        self.lengths = shape
        self.activation = activation
        self.output_activation=output_activation
        
        weights = []
        biases = []
        for i, length in enumerate(shape):
            layer_weights = []
            layer_biases = []
            for j in range(length):
                is_input = i == 0
                if not is_input:
                    neuron_weights = []
                    previous_layer_length = shape[i-1]
                    for k in range(previous_layer_length):
                        weight = generate_weight(min_weight=min_weight, max_weight=max_weight, closest_to_0=closest_to_0)
                        neuron_weights.append(weight)

                    layer_biases.append(bias)
                    layer_weights.append(neuron_weights)

            weights.append(layer_weights)
            biases.append(layer_biases)
        
        self.weights = weights
        self.biases = biases


if __name__ == '__main__':
    main()