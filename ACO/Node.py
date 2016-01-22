import random



class Node:

    def __init__(self, idx=None, country=None, city=None, lat=None, lon=None):
        self.idx = idx

        self.edges = []

        self.country = country
        self.city= city
        self.latitude = float(lat)
        self.longitude = float(lon)


    def roulette_wheel_simple(self):
        pass

    def roulette_wheel(self, visited_edges, start_node):

        # Find all edges ant has visited
        visited_nodes = [edge.second for edge in visited_edges]

        # Find viable edges for next walk
        # Restrictions:
        # * edge destination must not be in the visited_node list
        # * edge destination must not be the start_node
        viable_edges = [edge for edge in self.edges if not edge.second in visited_nodes and edge.second != start_node]

        # If no viable edge exists
        if not viable_edges:
            viable_edges = [edge for edge in self.edges if not edge.second in visited_nodes] # TODO - why not set start_node edges?

        # Sum all pheromones in viable edges
        all_pheromones = sum([edge.pheromones for edge in viable_edges])

        ######################################################
        ##
        ## Start roulette
        ##
        ######################################################

        # Get random number between 0 and all_pheromones
        random_number = random.uniform(0, all_pheromones)

        s = 0 # Roulette value
        i = 0 # Index of selected edge

        selected_edge = viable_edges[i]

        # Select edge while s is below the random generated number
        while s <= random_number:
            selected_edge = selected_edge[i]
            s += selected_edge.pheromones
            i += 1
        return selected_edge


    def __repr__(self):
        return str(self.idx)