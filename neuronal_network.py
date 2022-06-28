import random as r
from numpy import array
from classes.enums import activation, cost_functions
from classes.interconnection import interconnection
from functions.execution import linear_function, activate
from functions.training import activation_derivative, cost_function, cost_function_derivative
from functions.evolution import get_variation, will_mutate

def main():
    nn = neuronal_network()
    shape = [3, 1]
    nn.generate_from_shape(shape)
    nn.new_neuron_chance = .9
    nn.layer_over_neuron_addition_chance = 1
    for i in range(50):
        nn.evolve()
    nn.save_to_file('./network.txt')
    nn.read_from_file('./network.txt')
    #nn.save_to_file('./network.txt')
    values = []
    for i in range(nn.shape[0]):
        values.append(r.randint(0, 20) + r.random())
    print(f'execution with random values: {nn.execute_network(values)}')
    print(nn.activation)
        
class neuronal_network:
    def __init__(self, mutation_max_variation=.17, mutation_chance=.27, evolution_new_neuron_chance=.1, evolution_layer_over_neuron_addition_chance=1/10, max_mutate_mutation_value_variation=.05, max_mutate_mutation_of_mutations_variation=.03):
        self.mutation_max_variation = mutation_max_variation
        self.mutation_chance = mutation_chance
        self.new_neuron_chance = evolution_new_neuron_chance
        self.max_mutate_mutation_variation = max_mutate_mutation_value_variation
        self.max_mutate_mutation_of_mutations_variation = max_mutate_mutation_of_mutations_variation
        self.layer_over_neuron_addition_chance = evolution_layer_over_neuron_addition_chance
        self.has_interconnected_neurons = False
    
    def execute_network(self, input_vals: list[float], return_all_values=False):
        """When the argument return_all_values is False(default value) this function returns the output layer output, else it returns a tuple containing all the linear function outputs from all the neurons grouped in lists of lists and all the neuron outputs grouped in lists of lists"""
        neuron_outputs = [input_vals]
        for i in range(self.shape[0]):
            self.interconnections[0][i].value = input_vals[i]
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
                linear = linear_function(previous_activations, self.weights[layer][neuron], self.biases[layer][neuron])
                for i, pos in enumerate(self.interconnections[layer][neuron].backward_connections_pos):
                    #print(pos)
                    #print(len(self.shape))
                    #print(self.shape[pos[0]])
                    #print()
                    connection = self.interconnections[pos[0]][pos[1]]
                    linear += connection.value * connection.weights[i]
                    
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
    
    def fit(self, X: list[list[float]], y: list[list[float]], cost_function_enum: cost_function, create_cost_line_plot=True):
        if self.has_interconnected_neurons:
            exit(Exception('Cannot train interconnected, by evolution, networks'))
    
    def train(self, X: float, y: float, cost_function_enum: cost_functions):
        # setup / execute network
        linear_functions, neuron_outputs = self.execute_network(X, return_all_values=True)
        output = neuron_outputs[-1]
        costs = []
        cost_derivatives = []
        
        # calculate cost and cost_derivatives
        for i, output in enumerate(output):
            costs.append(cost_function(output, y[i]))
            cost_derivatives.append(cost_function_derivative(output, y[i], cost_function_enum))
        
        # get cost mean, informative output
        total_cost = 0
        for cost in costs:
            total_cost += cost
        total_cost /= len(costs)
        
        # set-up principal output
        bias_grads = []
        weight_grads = []
        
        # set-up variables
        layer_gradients = cost_derivatives
        previous_layer_gradients = layer_gradients
        for layer_i in range(len(self.shape) - 1, -1, -1):
            # create layer-lists for principal output
            bias_grads.append([])
            weight_grads.append([])
            
            # pass previous layer
            layer_gradients = previous_layer_gradients
            
            # select current activation function
            activation_function = None
            if layer_i == len(self.shape) - 1:
                activation_function = self.output_activation
            else:
                activation_function = self.activation
                
            previous_layer_gradients = []
            for neuron_i in range(self.shape[layer_i]):
                previous_layer_gradients.append(0)
                
            for neuron_i in range(self.shape[layer_i]):
                # calculate gradient passed activation function, &L(loss) / &A(activation)
                layer_gradients[neuron_i] *= activation_derivative(linear_functions[layer_i][neuron_i], activation_function)
                
                neuron_weights_grads = []
                
                for previous_layer_index in range(self.shape[layer_i - 1]):
                    # calculate weight gradient &L(loss) / &W(weight)
                    neuron_weights_grads.append(layer_gradients[neuron_i] * neuron_outputs[layer_i - 1][neuron_i][previous_layer_index])
                    # calculate previous layer gradients &L(loss) / &X(Previous layer activation)
                    previous_layer_gradients[neuron_i] += layer_gradients[neuron_i] * self.weights[layer_i][neuron_i][previous_layer_index]
                    
                weight_grads[layer_i].append(neuron_weights_grads)
                
                # save the calculated bias grad &L(loss) / &B(bias)
                bias_grads[layer_i].append(layer_gradients[neuron_i])
    
    def evolve(self):
        self.mutation_chance += get_variation(self.max_mutate_mutation_variation) * will_mutate(self.mutation_chance)

        self.max_mutate_mutation_variation += get_variation(self.max_mutate_mutation_of_mutations_variation) * will_mutate(self.mutation_chance)
        
        self.mutation_max_variation += get_variation(self.max_mutate_mutation_variation) * will_mutate(self.mutation_chance)
        
        self.layer_over_neuron_addition_chance += get_variation(self.max_mutate_mutation_variation) * will_mutate(self.mutation_chance)
        
        self.new_neuron_chance += get_variation(self.mutation_max_variation) * will_mutate(self.mutation_chance)
        
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
                for k in range(len(interconnection.weights)):
                    if will_mutate(self.mutation_chance):
                        self.interconnections[i][j].weights[k] += get_variation(self.mutation_max_variation)
             
        if self.new_neuron_chance > r.random():
            if r.random() > self.layer_over_neuron_addition_chance and len(self.shape) > 2:
                self.add_new_neuron()
            else:
                self.add_new_layer()
            
    def add_new_neuron(self):
        layer_count = len(self.shape)
        layer_index = r.randint(1, layer_count - 2)
        
        # at each neuron from layer_index + 1 add a new weight at the end
        for neuron_index, neuron_weights in enumerate(self.weights[layer_index + 1]):
            self.weights[layer_index + 1][neuron_index].append(generate_weight())
        
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
            
    def add_new_layer(self):
        layer_count = len(self.shape)
        layer_insertion_index = r.randint(1, layer_count - 1)
        
        # update interconnection positions
        for layer_index, layer_length in enumerate(self.shape):
            for neuron_index in range(layer_length):
                current_interconnection = self.interconnections[layer_index][neuron_index]
                
                for connection_i in range(len(current_interconnection.weights)):
                    if current_interconnection.front_connections_pos[connection_i][0] > len(self.shape):
                        print('error')
                    
                    # update interconnection's backwardly connected neuron position
                    current_backward_connected_layer = current_interconnection.backward_connections_pos[connection_i][0]
                    self.interconnections[layer_index][neuron_index].backward_connections_pos[connection_i][0] += current_backward_connected_layer >= layer_insertion_index
                    
                    # update front connected interconnections of previous layer
                    current_front_connected_layer = current_interconnection.front_connections_pos[connection_i][0]
                    self.interconnections[layer_index][neuron_index].front_connections_pos[connection_i][0] += current_front_connected_layer >= layer_insertion_index
        
        
        # add interconnection weights and positions
        for neuron_index, neuron_weights in enumerate(self.weights[layer_insertion_index]):
            for weight_index, weight in enumerate(neuron_weights):
                # connected neuron weight update
                self.interconnections[layer_insertion_index - 1][weight_index].weights.append(weight)
                
                # connect insertion index layer with previous layer
                self.interconnections[layer_insertion_index][neuron_index].backward_connections_pos.append([layer_insertion_index - 1, weight_index])
        
        # connect previous layer with insertion index layer
        previous_layer_length = self.shape[layer_insertion_index - 1]
        for weight_i in range(previous_layer_length):
            for neuron_i in range(self.shape[layer_insertion_index]):
                self.interconnections[layer_insertion_index - 1][weight_i].front_connections_pos.append([layer_insertion_index + 1, neuron_i])
        self.has_interconnected_neurons = True


        # update weights in the next layer of the inserted layer
        for neuron_index in range(self.shape[layer_insertion_index]):
            self.weights[layer_insertion_index][neuron_index] = [generate_weight()]
        
        
        # insert bias at layer insertion index
        self.biases.insert(layer_insertion_index, [1])
        
        # insert weights at layer insertion index
        neuron_weights = []
        for i in range(self.shape[layer_insertion_index - 1]):
            neuron_weights.append(generate_weight())
        self.weights.insert(layer_insertion_index, [neuron_weights])
        
        # insert an interconnection at layer insertion index
        self.interconnections.insert(layer_insertion_index, [interconnection()])
        
        # update shape
        self.shape.insert(layer_insertion_index, 1)
        
    def have_child(self, max_childs: int, min_childs: int, network_score: float, max_score: float, mutate_child_networks=True):
        output = []
        child_count = int(network_score / max_score * max_childs)
        child_count += (min_childs - child_count) * child_count < min_childs
        for i in range(child_count):
            parent_copy = neuronal_network(mutation_max_variation=self.mutation_max_variation, mutation_chance=self.mutation_chance, evolution_new_neuron_chance=self.new_neuron_chance)
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
        output += f'add new neuron evolution chance: {self.new_neuron_chance}\n'
        output += f'max mutate mutation value variation: {self.max_mutate_mutation_variation}\n'
        output += f'max mutate mutation of mutations variation: {self.max_mutate_mutation_of_mutations_variation}\n'
        output += f'layer_over_neuron_addition_chance: {self.layer_over_neuron_addition_chance}\n'
        
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
        self.has_interconnected_neurons = lines[4].removeprefix('has interconnections: ') == 'True'
        self.new_neuron_chance = float(lines[5].removeprefix('add new neuron evolution chance: '))
        self.max_mutate_mutation_variation = float(lines[6].removeprefix('max mutate mutation value variation: '))
        self.max_mutate_mutation_of_mutations_variation = float(lines[7].removeprefix('max mutate mutation of mutations variation: '))
        self.layer_over_neuron_addition_chance = float(lines[8].removeprefix('layer_over_neuron_addition_chance: '))
        
        biases_str = string.split('---')[1].removeprefix('\n').removesuffix('\n')
        network_biases = biases_str.split('\n')
        biases = [[]]
        self.shape = []
        for i, layer in enumerate(network_biases):
            layer_biases_str = layer.split(',')
            layer_biases = []
            for j, bias in enumerate(layer_biases_str):
                layer_biases.append(float(bias))
            biases.append(layer_biases)
            self.shape.append(len(layer_biases))
        self.biases = biases
        
        weights_str = string.split('___')[1].removeprefix('\n').removesuffix('\n')
        weights_str = weights_str.split('\n')
        weights = [[]]
        for i, layer in enumerate(weights_str):
            layer_weights_str = layer.split('/')
            layer_weights = []
            for j, neuron_weights_str in enumerate(layer_weights_str):
                neuron_weights = []
                splitted_neuron_weights_str = neuron_weights_str.split(',')
                for k, weight in enumerate(splitted_neuron_weights_str):
                    neuron_weights.append(float(weight))
                layer_weights.append(neuron_weights)
            weights.append(layer_weights)
        self.weights = weights
        self.shape.insert(0, len(weights[1][0]))
            
        connections_str = string.split('===')[1].removeprefix('\n').removesuffix('\n').split('\n')
        interconnections = []
        for i, layer_connections_str in enumerate(connections_str):
            layer_connections = []
            for j, connection_str in enumerate(layer_connections_str.split('/')):
                current_interconnection = interconnection()
                current_interconnection.from_str(connection_str)
                layer_connections.append(current_interconnection)
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
            
    def generate_from_shape(self, shape : list, activation=activation.sine, output_activation=None, bias=1, min_weight=-1.5, max_weight=1.5, generated_weight_closest_to_0=0.37) -> None:
        """shape: list containing the number of neurons at each layer, including input network. min_weight must be negative and max_weight negative for a correct functioning
        output_activation: if set to none it will use activation as activation function"""
        self.shape = shape
        self.activation = activation
        if output_activation != None:
            self.output_activation=output_activation
        else:
            self.output_activation=activation
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
        self.new_neuron_chance = network.new_neurons_chance
        self.max_mutate_mutation_variation = network.max_mutate_mutation_value_variation
        self.max_mutate_mutation_of_mutations_variation = network.max_mutate_mutation_of_mutations_variation

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

if __name__ == '__main__':
    main()