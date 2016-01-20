



class Edge:

    def __init__(self, first, second, cost, max_pheromones=100000):
        self.first = first
        self.second = second
        self.cost = cost
        self.pheromones = 1

        self.MAX_PHEROMONES = max_pheromones
        self.MIN_PHEROMONES = 0

    def check_pheromones(self):

        if self.pheromones > self.MAX_PHEROMONES:
            self.pheromones = self.MAX_PHEROMONES

        if self.pheromones < self.MIN_PHEROMONES:
            self.pheromones = self.MIN_PHEROMONES


    def __repr__(self):
        return self.first.name + "--(" + str(self.cost) + ")--" + self.second.name
