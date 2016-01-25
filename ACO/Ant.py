from ACO import ACO
import numpy as np
import random
import math
import time

class ANT:
    def __init__(self, start_node, nodes, edges, edges_pheromones, MAX_COST=1000, MAX_STEPS=12, MAX_PHEROMONES=1000, MIN_PHEROMONES=1, REQUIRED_STEPS=None, target_node=None):

        self.MAX_COST=MAX_COST

        # Start node of the ANT
        self.node = start_node

        self.MAX_STEPS = MAX_STEPS
        self.REQUIRED_STEPS = REQUIRED_STEPS
        self.MIN_PHEROMONES = MIN_PHEROMONES
        self.MAX_PHEROMONES = MAX_PHEROMONES
        self.target_node = target_node

        self.step = 0

        # Dataset
        self.nodes = nodes
        self.edges = edges
        self.edges_pheromones = edges_pheromones

        # Edges the ANT already used
        #
        self.possible_edges = {
            self.node.idx: self.edges[self.node.idx]
        }

        self.visited_nodes = set()

        # Add start node to visited nodes
        self.visited_nodes.add(self.node.idx)

        self.visited_edges = []

    def current_viable_edges(self):
        viable_edges = [(self.node.idx, y) for y in range(len(self.possible_edges[self.node.idx])) if y not in self.visited_nodes and y != self.target_node.idx]
        return viable_edges


    def edge_cost_sum(self):
        return sum([self.edges[idxs[0]][idxs[1]] for idxs in self.visited_edges])


    def roulette_wheel(self):

        viable_edges = self.current_viable_edges()

        sum_pheromones = sum([self.edges_pheromones[viable_edge[0]][viable_edge[1]] for viable_edge in viable_edges])


        random_number = random.uniform(0, sum_pheromones)

        roulette_sum = 0
        for edge in viable_edges:
            roulette_sum += self.edges_pheromones[edge[0], edge[1]]

            if roulette_sum >= random_number:
                return edge





    def walk(self):

        while self.node != self.target_node:

            # Ant walks onto a edge
            current_edge = self.roulette_wheel()

            if current_edge is None:
                current_edge = (self.node.idx, self.target_node.idx)
            elif self.MAX_STEPS != None and self.MAX_STEPS -1 == self.step:
                current_edge = (self.node.idx, self.target_node.idx)

            self.visited_nodes.add(current_edge[1])
            self.visited_edges.append(current_edge)
            self.node = self.nodes[current_edge[1]]

            try:
                self.possible_edges[self.node.idx]
            except:
                self.possible_edges[self.node.idx] = self.edges[self.node.idx]

            self.step += 1

        return self.node

    def pheromones(self):
        #  x ---- 200 ---- x ----- 400 ----- x ----- 100 ---- x ----- 2000 ---- x

        current_pheromones = 0
        score = 1000**(1-float(self.edge_cost_sum())/self.MAX_COST) # Score function


        for idx, v_edge in enumerate(self.visited_edges):

            current_pheromones = current_pheromones + self.edges_pheromones[v_edge[0]][v_edge[1]]
            #score = ((1 - .90) * self.edges_pheromones[v_edge[0]][v_edge[1]]) + (1 / self.edge_cost_sum())
            #print(score)|

            score = 1 * math.pow(self.edges_pheromones[v_edge[0]][v_edge[1]],  2) + 0.5
            #score = 1*math.pow(((1 - .90) * self.edges_pheromones[v_edge[0]][v_edge[1]]) + (1 / self.edge_cost_sum()),2)+0.5

            self.edges_pheromones[v_edge[0]][v_edge[1]] = self.edges_pheromones[v_edge[0]][v_edge[1]] + score
            #print(self.edges_pheromones[v_edge[0]][v_edge[1]])


