import Download
import codecs
from ACO.Ant import ANT
from ACO.Node import Node
from ACO.ACO import ACO
from ACO.Edge import Edge
from ACO import CityFilter
import random
import numpy as np
import ACO.CityFilter
import ACO.CUDA
import time

world_cities_path = "./Download/worldcitiespop.txt.gz"
world_cities_txt_path = "./Download/worldcitiespop.txt"

# Download world_cities
#world_cities_path = Download.download("http://download.maxmind.com/download/worldcities/worldcitiespop.txt.gz", "./Download/")

# Decompress
#world_cities_txt_path = Download.gz(world_cities_path)

# Download country_codes
country_codes_path = Download.download("http://data.okfn.org/data/core/country-list/r/data.csv", "./Download/")


def load_country_codes():
    country_codes = {}
    with open(country_codes_path) as file:
        for row in file.readlines():
            row = row.replace("\n", "").replace("\"","")
            split = row.rsplit(',', 1)
            #country_codes[row[1]] = row[0]
    return country_codes

def load_world_cities(loc=["*"]):
    nodes = []
    nodes_latitude = []
    nodes_longitude = []

    with codecs.open(world_cities_txt_path, "r",encoding='utf-8', errors='ignore') as file:
        items = file.readlines()[1:]

        i=0
        for row in items:
            # Country, City, AccentCity, Region, Population, Latitude, Longitude
            split = row.split(",")
            c_code = str(split[0])
            city = str(split[1])
            lat = str(split[5])
            lon = str(split[6].replace("\n",""))

            if "*" in loc or c_code in loc:
                nodes.append(Node(idx=i, country=c_code, city=city, lat=lat, lon=lon))
                nodes_latitude.append(lat)
                nodes_longitude.append(lon)
                i += 1

    return nodes, np.array(nodes_latitude, dtype=np.float32), np.array(nodes_longitude, dtype=np.float32)

# Load country codes
start = time.time()
country_codes = load_country_codes()
print("[+{0}] Loaded country_code converter!".format(round(time.time() - start, 2)))

# Load Nodes
start = time.time()
nodes, nodes_latitude, nodes_longitude = load_world_cities([ "no", "se", "dk"])
print("[+{0}] Parsed {1} nodes.".format(round(time.time() - start, 2), len(nodes)))

# Create edges
start = time.time()
edges = ACO.CUDA.create_distance_matrix(nodes_latitude, nodes_longitude)
print("[+{0}] Generated {1} edges using GPU".format(round(time.time() - start, 2), len(edges) ** 2))

# Create edge_pheromones
start = time.time()
edges_pheromones = np.ones((nodes_latitude.shape[0], nodes_longitude.shape[0]), dtype=np.int32)
print("[+{0}] Created pheromones map".format(round(time.time() - start, 2)))



#distance = ACO.CityFilter.calculate_distance_in_metres(nodes_latitude[12000], nodes_longitude[12000], nodes_latitude[1], nodes_longitude[1])
#print(distance)
#print(edges[12000][1])


MAX_COST = np.sum(edges) # TODO , some nan values. BUT WHERE?

start_node = nodes[0]

target_node = nodes[5153]

for i in range(1000000):
    ant = ANT(start_node, nodes, edges, edges_pheromones, MAX_COST=MAX_COST, target_node=target_node)
    goal = ant.walk()
    ant.pheromones()

    print("[{0}]: Edges: {2}, Route: {1}".format(i, sum([edges[item[0]][item[1]] for item in ant.visited_edges]), len(ant.visited_edges)))

