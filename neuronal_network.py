import random as r
from functions import generate_weight, linear_function, activate, derivative, get_variation, will_mutate
from classes import activation, interconnection, interconnection_from_string

def main():
    nn = neuronal_network()
    shape = [728, 1]
    nn.generate_from_shape(shape)
    nn.save_to_file('./network.txt')
    nn.read_from_file('./network.txt')
    print(nn.activation)
        
class neuronal_network:
    def __init__(self, mutation_max_variation=.17, mutation_chance=.27, evolution_new_neurons_chance=.1, max_mutate_mutation_value_variation=.05, max_mutate_mutation_of_mutations_variation=.03):
        self.mutation_max_variation = mutation_max_variation
        self.mutation_chance = mutation_chance
        self.new_neurons_chance = evolution_new_neurons_chance
        self.max_mutate_mutation_variation = max_mutate_mutation_value_variation
        self.max_mutate_mutation_of_mutations_variation = max_mutate_mutation_of_mutations_variation
    
    def execute_network(self, input_vals: list[float], return_all_values=False):
        """When the argument return_all_values is False(default value) this function returns the output layer output, else it returns a tuple containing all the linear function outputs from all the neurons grouped in lists of lists and all the neuron outputs grouped in lists of lists"""
        neuron_outputs = [input_vals]
        for i in range(self.shape[0]):
            self.interconnections[0][i] = input_vals[i]
        linear_functions = []
        for layer in range(1, len(self.shape)):
            layer_output = []
            layer_linear_functions = []
            
            is_output_layer = layer == len(self.shape) - 1
            if not is_output_layer:
                activation_function = self.activation
            else:
                activation_function = self.output_activation
                
            for neuron in range(self.shape[layer]):
                previous_activations = neuron_outputs[-1]
                linear = linear_function(previous_activations, self.weights[layer][neuron])
                for i, pos in enumerate(self.interconnections[layer][neuron].backward_connected_neuron_pos):
                    connection = self.interconnections[pos[0]][pos[1]]
                    
                    linear += connection.value * connection.weights[[layer, neuron]]
                activated_value = activate(linear, activation_function)
                self.interconnections[layer][neuron].value = activated_value
                
                layer_output.append(activated_value)
                
                layer_linear_functions.append(linear)
            
            linear_functions.append(layer_linear_functions)
            neuron_outputs.append(layer_output)
        
        if not return_all_values:
            return neuron_outputs[-1]
        else:
            return (linear_functions, neuron_outputs)
    
    def evolve(self):
        if will_mutate(self.mutation_chance):
            self.mutation_chance += get_variation(self.max_mutate_mutation_variation)
        
        if will_mutate(self.mutation_chance):
            self.max_mutate_mutation_variation += get_variation(self.max_mutate_mutation_of_mutations_variation)
            
        if will_mutate(self.mutation_chance):
            self.mutation_max_variation += get_variation(self.max_mutate_mutation_variation)
            
        if will_mutate(self.mutation_chance):
            self.add_new_neuron_chance += get_variation(self.mutation_max_variation)
        
        for i, layer_weights in enumerate(self.weights):
            for j, neuron_weights in enumerate(layer_weights):
                for k, weight in enumerate(neuron_weights):
                    if will_mutate(self.mutation_chance):
                        self.weights[i][j][k] += get_variation(self.mutation_max_variation)
        
        for i, layer_biases in enumerate(self.biases):
            for j, bias in enumerate(layer_biases):
                if will_mutate(self.mutation_chance):
                    self.biases[i][j] += get_variation(self.mutation_max_variation)
                
        for i, layer_interconnections in enumerate(self.interconnections):
            for j, interconnection in enumerate(layer_interconnections):
                for k in range(len(interconnection)):
                    if will_mutate(self.mutation_chance):
                        self.interconnections[i][j].front_weights[k] += get_variation(self.mutation_max_variation)
                    
        if r.random() > self.new_neurons_chance:
            self.add_new_neuron()
            
    def add_new_neuron(self):
        layer_over_neuron_addition_chance = 1 / 6
        
        layer_count = len(self.shape)
        if r.random() > layer_over_neuron_addition_chance and layer_count > 2: # add new neuron
            layer_index = r.randint(1, layer_count - 1)
            
            # at each neuron from layer_index + 1 add a new weight at the end
            for i, neuron_weights in enumerate(self.weights[layer_index + 1]):
                neuron_weights.append(generate_weight())
            
            # append the neuron with a list of weights with the previous layer length as its length
            weights = []
            previous_layer_length = self.shape[layer_index - 1]
            for i in range(previous_layer_length):
                weights.append(generate_weight())
            self.weights[layer_index].append(weights)
            
            # append a bias and an empty interconnection
            self.biases[layer_index].append(1)
            self.interconnections[layer_index].append(interconnection())
            
            # update shape
            self.shape[layer_index] += 1
            
        else: # add new layer with one neuron
            layer_insertion_index = r.randint(1, layer_count)
            
            # update interconnection positions
            for layer_index, layer in enumerate(len(self.shape)):
                for neuron_index in range(layer):
                    interconnection = self.interconnections[layer_index][neuron_index]
                    # update interconnection's backwardly connected neuron position
                    self.interconnections[layer_index][neuron_index].backward_connected_neuron_pos[0] += interconnection[0] >= layer_insertion_index
                    
                    # set forwardly connected interconnections of previous layer
                    self.interconnections[layer_index][neuron_index].forward_connected_neuron_pos[0] += interconnection[0] >= layer_insertion_index
            
            # set interconnection weights and positions
            for neuron_index, neuron in enumerate(self.weights[layer_insertion_index]):
                for weight_index, weight in enumerate(neuron):
                    # connected neuron weight update
                    self.interconnections[layer_insertion_index - 1][weight_index].weights.append(weight)
                    
                    # connect previous layer with insertion index layer
                    self.interconnections[layer_insertion_index - 1][weight_index].front_connected_neuron_pos.append([layer_insertion_index, neuron_index])
                    
                    # connect insertion index layer with previous layer
                    self.interconnections[layer_insertion_index][neuron_index].backward_connected_neurons_pos.append([layer_insertion_index - 1, weight_index])
            
            # update weights and bias in the next layer of the inserted layer
            for neuron_index in range(self.shape[layer_insertion_index]):
                self.weights[layer_insertion_index][neuron_index] = [generate_weight()]
            
            # insert weights at layer insertion index
            neuron_weights = []
            for i in range(self.shape[layer_insertion_index - 1]):
                neuron_weights.append(generate_weight())
            self.weights.insert(layer_insertion_index, [neuron_weights])
            
            # update shape
            self.shape.insert(layer_insertion_index, 1)
        
    def have_child(self, max_childs: float, network_score: float, max_score: float, mutate_child_networks=True):
        output = []
        child_count = int(network_score / max_score * max_childs)
        for i in range(child_count):
            parent_copy = neuronal_network(mutation_max_variation=self.mutation_max_variation, mutation_chance=self.mutation_chance, evolution_new_neurons_chance=self.new_neurons_chance)
            parent_copy.from_network(self)
            
            if mutate_child_networks:
                parent_copy.evolve()
            
            output.append(parent_copy)
        return output
    
    def __str__(self) -> str:
        activation = str(self.activation).removeprefix('activation.')
        output_activation = str(self.output_activation).removeprefix('activation.')
        output = f'activation: {activation}\noutput activation: {output_activation}\n'
        
        output += f'mutation max variation: {self.mutation_max_variation}\n' + f'mutation chance: {self.mutation_chance}\n'
        
        output += f'has interconnections: {self.has_interconnected_neurons}\n'
        output += f'add new neurons evolution chance: {self.new_neurons_chance}\n'
        output += f'max mutate mutation value variation: {self.max_mutate_mutation_variation}\n'
        output += f'max mutate mutation of mutations variation: {self.max_mutate_mutation_of_mutations_variation}\n'
        
        output += 'biases:\n---'
        for i, layer_biases in enumerate(self.biases):
            for j, bias in enumerate(layer_biases):
                output += f'{bias},'
            output = output.removesuffix(',')
            output += '\n'
        output += '---\n'

        output += 'weights:\n___'
        for i, layer_weights in enumerate(self.weights):
            for j, neuron_weights in enumerate(layer_weights):
                for k, weight in enumerate(neuron_weights):
                    output += f'{weight},'
                output = output.removesuffix(',')
                output += '/'
            output = output.removesuffix('/')
            output += '\n'
        output += '___\n'
        
        output += 'interconnections:\n===\n'
        for i, layer_connections in enumerate(self.interconnections):
            for j, connection in enumerate(layer_connections):
                output += f'{connection}/'
            output = output.removesuffix('/')
            output += '\n'
        output = output.removesuffix('\n')
        
        return output
    
    def from_str(self, string: str):
        lines = string.splitlines(False)
        self.activation = activation(lines[0].removeprefix('activation: '))
        self.output_activation = activation(lines[1].removeprefix('output activation: '))
        self.mutation_max_variation = float(lines[2].removeprefix('mutation max variation: '))
        self.mutation_chance = float(lines[3].removeprefix('mutation chance: '))
        self.has_interconnected_neurons = bool(lines[4].removeprefix('has interconnections: '))
        self.add_new_neuron_chance = float(lines[5].removeprefix('add new neurons evolution chance: '))
        self.max_mutate_mutation_variation = float(lines[6].removeprefix('max mutate mutation value variation: '))
        self.max_mutate_mutation_of_mutations_variation = float(lines[7].removeprefix('max mutate mutation of mutations variation: '))
        
        biases_str = string.split('---')[1]
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
        self.biases = biases
        
        weights_str = string.split('___')[1]
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
            
        connections_str = string.split('===\n')[1].split('\n')
        interconnections = []
        for i, layer_connections_str in enumerate(connections_str):
            layer_connections = []
            for j, connection_str in enumerate(layer_connections_str.split('/')):
                layer_connections.append(interconnection_from_string(connection_str))
            interconnections.append(layer_connections)
        
        self.interconnections = interconnections
    
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
            
    def generate_from_shape(self, shape : list, activation=activation.sine, output_activation=activation.sine, bias=1, min_weight=-1.5, max_weight=1.5, generated_weight_closest_to_0=0.37) -> None:
        """shape: list containing the number of neurons at each layer, including input network. min_weight must be negative and max_weight negative for a correct functioning"""
        self.lengths = shape
        self.activation = activation
        self.output_activation=output_activation
        self.has_interconnected_neurons = False
        
        weights = []
        biases = []
        interconnections = []
        for layer_index, length in enumerate(shape):
            layer_weights = []
            layer_biases = []
            layer_interconnections = []
            for neuron_index in range(length):
                layer_interconnections.append(interconnection())
                
                is_input = layer_index == 0
                if not is_input:
                    neuron_weights = []
                    previous_layer_length = shape[layer_index-1]
                    for weight_index in range(previous_layer_length):
                        weight = generate_weight(min_weight=min_weight, max_weight=max_weight, closest_to_0=generated_weight_closest_to_0)
                        neuron_weights.append(weight)

                    layer_biases.append(bias)
                    layer_weights.append(neuron_weights)

            weights.append(layer_weights)
            biases.append(layer_biases)
            interconnections.append(layer_interconnections)
        
        self.weights = weights
        self.biases = biases
        self.interconnections = interconnections
               
    def from_network(self, network) -> None:
        self.activation = network.activation
        self.output_activation = network.output_activation
        
        self.weights = network.weights
        self.biases = network.biases
        self.interconnections = network.interconnections
        
        self.mutation_max_variation = network.mutation_max_variation
        self.mutation_chance = network.mutation_chance
        self.has_interconnected_neurons = network.has_interconnected_neurons
        self.add_new_neuron_chance = network.add_new_neurons_chance
        self.max_mutate_mutation_variation = network.max_mutate_mutation_value_variation
        self.max_mutate_mutation_of_mutations_variation = network.max_mutate_mutation_of_mutations_variation

if __name__ == '__main__':
    main()