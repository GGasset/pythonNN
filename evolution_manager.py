import neuronal_network as nn

class evolution_manager:
    def __init__(self, starting_network_count: int, starting_shape: list[int]):
        self.networks = []
        for i in range(starting_network_count):
            network = nn.neuronal_network()
            network.generate_from_shape(starting_shape)
            self.networks.append(network)
        self.scores = {}
    
    def have_child(self, max_child_per_network)->None:
        '''All the networks must have a rating to properly work. Childs will automatically mutate.'''
        max_score = 0
        for i in self.scores.keys:
            max_score += (self.scores[i] - max_score) * self.scores[i] > max_score
        for i, network in enumerate(self.networks):
            childs = network.have_child(max_child_per_network, self.scores[i], max_score)
            for i, child in enumerate(childs):
                self.networks.append(child)
                
    def set_score(self, index: int, score: float)->None:
        self.scores[index] = score