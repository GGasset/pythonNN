class interconnection:
    def __init__(self, front_weights=[], front_connections_pos=[], backward_connections_pos=[]):
        self.weights = front_weights
        self.front_connections_pos = front_connections_pos
        self.backward_connections_pos = backward_connections_pos
        self.value = 0
        
    def __str__(self):
        output = ''
        for i, weight in enumerate(self.weights):
            output += f'{weight},'
        output = output.removesuffix(',')
        output += '+'
        
        for i, front_pos in enumerate(self.front_connections_pos):
            output += f'layer_i: {front_pos[0]} - neuron_i: {front_pos[1]},'
        output = output.removesuffix(',')
        output += '+'
        
        for i, back_pos in enumerate(self.backward_connections_pos):
            output += f'layer_i: {back_pos[0]} - neuron_i: {back_pos[1]},'
        output = output.removesuffix(',')
        
        return output
    
    def from_str(self, string: str) -> None:      
        if string == '++':
            self.weights = []
            self.front_connections_pos = []
            self.backward_connections_pos = []
            return
          
        string = string.replace('layer_i: ', '').replace('neuron_i: ', '').split('+')
        
        weights_str = string[0].split(',')
        self.weights = []
        for i, weight_str in enumerate(weights_str):
            self.weights.append(float(weight_str))
        
        front_positions_str = string[1].split(',')
        self.front_connections_pos = []
        for i, pos_str in enumerate(front_positions_str):
            splitted_pos_str = pos_str.split(' - ')
            position = [int(splitted_pos_str[0]), int(splitted_pos_str[1])]
            self.front_connections_pos.append(position)
        
        back_positions_str = string[2].split(',')
        self.backward_connections_pos = []
        for i, pos_str in enumerate(back_positions_str):
            splitted_pos_str = pos_str.split(' - ')
            position = [int(splitted_pos_str[0]), int(splitted_pos_str[1])]
            self.backward_connections_pos.append(position)