from ACO import ACO
import numpy as np
import random
import time

class ANT:
    def __init__(self, start_node, nodes, edges, edges_pheromones, MAX_COST=1000, MAX_STEPS=10, target_node=None):

        self.MAX_COST=MAX_COST

        # Start node of the ANT
        self.node = start_node

        self.MAX_STEPS = MAX_STEPS

        self.target_node = target_node

        self.step = 0

        # Dataset
        self.nodes = nodes
        self.edges = edges
        self.edges_pheromones = edges_pheromones

        # Edges the ANT already used
        self.viable_edges = {
            self.node.idx: self.edges[self.node.idx]
        }

        self.visited_nodes = set()

        # Add start node to visited nodes
        self.visited_nodes.add(self.node.idx)

        self.visited_edges = []

    def current_viable_edges(self):
        return self.viable_edges[self.node.idx]

    def edge_cost_sum(self):
        return sum([self.edges[idxs[0]][idxs[1]] for idxs in self.visited_edges])


    def roulette_wheel(self):
        # Sum all pheromones in viable edges
        sum_pheromones = sum([self.edges_pheromones[self.node.idx][index[0]] for index, x in np.ndenumerate(self.current_viable_edges())])

        ######################################################
        ##
        ## Start roulette
        ##
        ######################################################

        # Get random number between 0 and all_pheromones
        random_number = random.uniform(0, sum_pheromones)

        selected_edge = None # Should never return None

        while True:

            s = 0 # Roulette value
            i = 0 # Index of selected edge

            # Roulette until a edge is found
            while s <= random_number:
                selected_edge = (self.node.idx, i)
                s += self.edges_pheromones[selected_edge[0], selected_edge[1]]
                i += 1


            # Break out of loop if the selected edge is not visied before
            if selected_edge not in self.visited_edges:
                break

        return selected_edge

    def walk(self):


        #print("[ANT]: Starts walking from {0}".format(current_node.idx))

        #print("[ANT]: Has {0} viable edges from {1}".format(len(self.viable_edges[current_node.idx]), current_node.idx))


        #print("[ANT]: Walking on edge {0}...".format(current_edge))


        while len(self.visited_nodes) != len(self.nodes):
            # Ant walks onto a edge
            current_edge = self.roulette_wheel()
            self.visited_edges.append(current_edge)
            self.visited_nodes.add(current_edge[1])
            self.node = self.nodes[current_edge[1]]
            self.viable_edges[self.node.idx] = self.edges[self.node.idx]
            self.step += 1

            if self.target_node == self.node:
                break

        return self.node



    def pheromones(self):

        current_cost = self.edge_cost_sum()

        if current_cost < self.MAX_COST:
            score = 1000**(1 - float(current_cost) / self.MAX_COST)

            for edge in self.visited_edges:
                self.edges[edge[0]][edge[1]] += score




