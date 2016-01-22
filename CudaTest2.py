import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import codecs
import math
import ACO.CityFilter


world_cities_path = "./Download/worldcitiespop.txt.gz"
world_cities_txt_path = "./Download/worldcitiespop.txt"
def load_world_cities(loc=["*"]):
    nodes_latitude = []
    nodes_longitude = []

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
                nodes_latitude.append(lat)
                nodes_longitude.append(long)

    return np.array(nodes_latitude, dtype=np.float32), np.array(nodes_longitude, dtype=np.float32)

nodes_latitude, nodes_longitude = load_world_cities(["no"])
#########################################




#float *a, float *b, int len
mod = SourceModule("""
__global__ void calculate_distance(float *dest, float *a, float *b, int node_length)
{

    // Indexes of arrays
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int idy = threadIdx.y + blockDim.y * blockIdx.y;

    // Current 2d array position
    const int pos = idx + node_length * idy;

    // Get latitude and longitude
    const float lat_1 = a[idx];
    const float lat_2 = a[idy];
    const float lon_1 = b[idx];
    const float lon_2 = b[idy];

    /*const float lat_1 = a[0];
    const float lat_2 = a[1];
    const float lon_1 = b[0];
    const float lon_2 = b[1];*/

    const int r = 6371000;

    // Start calculation
    float phi_1 = (lat_1 * M_PI / 180.0);
    float phi_2 = (lat_2 * M_PI / 180.0);
    float delta_phi = ((lat_2 - lat_1) * M_PI / 180.0);
    float delta_lambda = ((lon_2 - lon_1) * M_PI / 180.0);

    float ang = (pow(sin(delta_phi / 2), 2)) + (cos(phi_1) * cos(phi_2) * pow(sin(delta_lambda), 2));
    float c = 2 * atan2(sqrt(ang), sqrt(1 - ang));
    float d = r * c;

    // Insert result
    dest[pos] = d;





}
""")

#Only main Device
MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
BLOCK_SIZE = int(math.sqrt(MAX_THREADS_PER_BLOCK))

rows = nodes_latitude
cols = nodes_longitude

dx, mx = divmod(cols.shape[0], BLOCK_SIZE)
dy, my = divmod(rows.shape[0], BLOCK_SIZE)

gdim = ( (dx + (mx>0)), (dy + (my>0)) )

edges = np.zeros((rows.shape[0], cols.shape[0]))
edges = edges.astype(np.float32)


diag_kernel = mod.get_function("diag_kernel")

#edges = np.zeros((len(nodes_latitude),len(nodes_longitude)), dtype=np.float32)
edges_bytes = edges.size * edges.dtype.itemsize
edges_gpu = drv.mem_alloc(edges_bytes)


drv.memcpy_htod(edges_gpu, edges)
diag_kernel(edges_gpu, drv.In(nodes_latitude),drv.In(nodes_longitude), np.int32(len(nodes_latitude)), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
drv.memcpy_dtoh(edges, edges_gpu)




### TEST HERE
cpu = ACO.CityFilter.calculate_distance_in_metres(nodes_latitude[0], nodes_longitude[0], nodes_latitude[1], nodes_longitude[1])
print("--CPU--")
print(cpu)
print("--GPU---")
print(edges[0][1])
#print("Lat 0: " + str(nodes_latitude[0]))


