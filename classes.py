from enum import Enum


class activation(Enum):
    sigmoid = 'sigmoid'
    sine = 'sine'
    none = 'none'

class interconnection:
    def __init__(self, front_weights={}, front_connections_pos=[], backward_connected_neurons_pos=[]):
        self.weights = front_weights
        self.front_connected_neurons_pos = front_connections_pos
        self.backward_connected_neuron_pos = backward_connected_neurons_pos
        self.value = 0
        
    def __str__(self):
        output = f'{self.weights}+{self.front_connected_neurons_pos}+{self.backward_connected_neuron_pos}'
        return output
    

    def from_string(self, string):
        string = string.split('+')
        return interconnection(front_weights=dict(string[0]), front_connections_pos=list(string[1]), backward_connected_neurons_pos=list(string[2]))