import random as r

def will_mutate(probability: float):
    return r.random() <= probability

def get_variation(max_variation:float):
    negative = r.randint(0, 1)
    negative -= 1 * (negative == 0)
    return r.random() * max_variation * negative