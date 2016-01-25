import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
import math




def create_distance_matrix(nodes_latitude, nodes_longitude):

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

    //const int r = 6371000;
    const int r = 1; //6371; // In KM
    //const int r = 6371; // In KM

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

    distance_kernel_func = mod.get_function("calculate_distance")


    #######################################
    ##
    ## Determine:
    ## * MAX THREADS
    ## * BLOCK_SIZE
    ## * NUMBER OF DIMENSIONS
    ##
    MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
    BLOCK_SIZE = int(math.sqrt(MAX_THREADS_PER_BLOCK))

    rows = nodes_latitude
    cols = nodes_longitude

    dx, mx = divmod(cols.shape[0], BLOCK_SIZE)
    dy, my = divmod(rows.shape[0], BLOCK_SIZE)

    gdim = ( (dx + (mx>0)), (dy + (my>0)) )

    ########################################
    ##
    ## Create edge matrix
    ##
    print("Creating {0}x{1} matrix".format(rows.shape[0], cols.shape[0]))
    edges = np.zeros((rows.shape[0], cols.shape[0]))
    print("Array is {0} mb".format((edges.nbytes / 1000) / 1000))
    edges = edges.astype(np.float32)
    edges_bytes = edges.size * edges.dtype.itemsize
    edges_gpu = drv.mem_alloc(edges_bytes)

    ########################################
    ##
    ## Execute on GPU
    ##
    drv.memcpy_htod(edges_gpu, edges)
    distance_kernel_func(edges_gpu, drv.In(nodes_latitude),drv.In(nodes_longitude), np.int32(len(nodes_latitude)), block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=gdim)
    drv.memcpy_dtoh(edges, edges_gpu)

    return edges
