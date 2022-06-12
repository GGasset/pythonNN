import neuronal_network as nn

class evolution_manager:
    def __init__(self, starting_network_count: int, starting_shape: list[int]):
        self.networks = []
        for i in range(starting_network_count):
            network = nn.neuronal_network()
            network.generate_from_shape(starting_shape)
            self.networks.append(network)
        self.ratings = []
    
    def have_child(self, max_child_per_network)->None:
        '''All the networks must have a rating to properly work. Childs will automatically mutate.'''
        max_rating = 0
        for i in self.ratings.keys:
            max_rating += (self.ratings[i] - max_rating) * self.ratings[i] > max_rating
        for i, network in enumerate(self.networks):
            childs = network.have_child(max_child_per_network, self.ratings[i], max_rating)
            for i, child in enumerate(childs):
                self.networks.append(child)
                
    def set_score(self, index: int, score: float)->None:
        self.ratings[index] = score
        
    def get_unrated_networks(self) -> list[nn.neuronal_network]:
        return self.networks[len(self.ratings):]
    
    def rate_networks(self, ratings: list[float]):
        for i, rating in enumerate(ratings):
            self.ratings.append(rating)