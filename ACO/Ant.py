from ACO import ACO

class ANT:
    def __init__(self, start_node):


        self.node = start_node
        self.visited_edges = []


    def walk(self):
        current_node = self.node
        current_edge = None



        while not ACO.ACO.has_visited_all(self.visited_edges):
            current_edge = current_node.roulette_wheel(self.visited_edges, self.node)
            current_node = current_edge.second
            self.visited_edges.append(current_edge)


    def pheromones(self):

        current_cost = ACO.ACO.get_sum(self.visited_edges)

        if current_cost < ACO.MAX_COST:
            score = 1000**(1 - float(current_cost) / ACO.MAX_COST)

            for edge in self.visited_edges:
                edge.pheromones += score