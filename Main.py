import Download
import codecs
from ACO.Ant import ANT
from ACO.Node import Node
from ACO.Edge import Edge
from ACO import CityFilter
import random

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
    world_cities = []

    with codecs.open(world_cities_txt_path, "r",encoding='utf-8', errors='ignore') as file:
        items = file.readlines()[1:]
        for row in items:
            # Country, City, AccentCity, Region, Population, Latitude, Longitude
            split = row.split(",")
            c_code = str(split[0])
            city = str(split[1])
            lat = str(split[5])
            long = str(split[6].replace("\n",""))

            if "*" in loc or c_code in loc:
                world_cities.append([c_code, city, lat, long])

    return world_cities



country_codes = load_country_codes()
world_cities = load_world_cities(["no"])


nodes = [Node("NA", country="NA", latitude=x[2], longitude=x[3]) for x in world_cities]
nodes = CityFilter.strip(nodes, 1000)

print("Created {0} nodes.".format(len(nodes)))


"""
edges = [Node(j, k) for j in nodes for k in nodes if j != k]
print("Create {0} edges.".format(len(edges)))
"""



"""
for i in range(100000):
    ant = ANT()
    ant.walk(a)
    ant.pheromones()
"""