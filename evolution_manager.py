import neuronal_network as nn

def main():
    manager = evolution_manager(100, [625, 1])
    #manager = from_str(str(manager))
    with open('networks.txt', 'w') as f:
        f.write(str(manager))

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
            
    def __str__(self) -> str:
        output = ''
        for i, network in enumerate(self.networks):
            output += ''
            rating_str = ''
            if i < len(self.ratings):
                rating_str += str(self.ratings[i])
            else:
                rating_str += 'None'
            output += f'\n,,,\nnetwork rating: {rating_str}\n<<<\n{str(network)}'
        output = output.removeprefix('\n,,,\n')
        return output
    
    def from_str(self, string: str):
        networks_str = string.split('\n,,,\n')
        output = evolution_manager(0, [])
        for i, network_str in enumerate(networks_str):
            rating_and_network_str = network_str.split('\n<<<\n')
            rating_str = rating_and_network_str[0].removeprefix('network rating: ')
            
            network = nn.neuronal_network()
            network.from_str(rating_and_network_str[1])
            
            if rating_str != 'None':
                self.ratings.append(int(rating_str))
            self.networks.append(network)
        return output
    
    def read_from_file(self, path: str):
        path += '.txt' * path.__contains__('.txt')
        text = ''
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                text += line
        self.from_str(text)

        
if __name__ == '__main__':
    main()