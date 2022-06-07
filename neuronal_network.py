import random as r
from enum import Enum

def main():
    nn = neuronal_network()
    nn.generate_from_shape([728, 500, 100, 40, 1])
    nn.save_to_file('./network.txt')

def generate_weight(min_weight=-1.5, max_weight=1.5, closest_to_0=0.37):
    """min_weight must be negative and max_weight negative for a correct functioning"""
    is_below_threshold = True
    closest_to_0 = abs(closest_to_0)
    while is_below_threshold:
        negative = r.randint(0, 1)
        weight = r.random() * (min_weight * negative + max_weight * (negative == 0))
        is_below_threshold = weight < closest_to_0
        #weight += (closest_to_0 - weight) * int(weight < closest_to_0)
    weight *= (-1 * negative) + (1 * negative == 0)
    return weight

class activation(Enum):
    sigmoid = 'sigmoid'
    sine = 'sine'
    none = 'none'

class neuronal_network:
    def __init__(self):
        pass
    
    def __str__(self) -> str:
        output = f'activation: {self.activation}\n'
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
        output += '___'
        
        return output
    
    def save_to_file(self, relative_path : str) -> None:
        if not relative_path.__contains__('.txt'):
            relative_path += '.txt'
        with open(relative_path, 'w') as f:
            f.write(self.__str__())
            
    def generate_from_shape(self, shape : list, activation=activation.sine, bias = 1, min_weight=-1.5, max_weight=1.5, closest_to_0=0.37) -> None:
        """shape: list containing the number of neurons at each layer, including input network. min_weight must be negative and max_weight negative for a correct functioning"""
        self.lengths = shape
        self.activation = activation
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