# Sample source code from the Tutorial Introduction in the documentation.




import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

nodes_latitude = numpy.random.rand(10).astype(numpy.float32)
nodes_longitude = numpy.random.rand(10).astype(numpy.float32)
edges = numpy.zeros(shape=(10, 10), dtype=float).astype(numpy.float32)

blocks = 64
block_size = 128
nbr_values = blocks * block_size
print(nbr_values)
######################
# SourceModele SECTION
#


# We write the C code and the indexing and we have lots of control

mod = SourceModule("""
__global__ void gpu_test(float *dest, float *a, float *b, int n_iter)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for(int n = 0; n < n_iter; n++) {
    dest[n] = n;
  }
}
""")

mod = SourceModule("""
__global__ void gpu_test(float *dest, float *a, float *b, int num_nodes)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;

  for(int n = 0; n < num_nodes; n++) {

  }
  dest[(int)i][0] = (int)i;
}
""")

#gpu_test = mod.get_function("gpu_test")
#gpu_test(cuda.Out(edges), cuda.In(nodes_latitude), cuda.In(nodes_longitude), numpy.int32(10), grid=(blocks,1), block=(block_size,1,1) )


gpusin = mod.get_function("gpu_test")

gpusin(cuda.Out(edges), cuda.In(nodes_latitude), cuda.In(nodes_longitude), numpy.int32(len(nodes_latitude)), grid=(blocks,1), block=(block_size,1,1) )


for i in edges:
    print(i)




exit(0)





a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print "original array:"
print a
print "doubled with kernel:"
print a_doubled

# alternate kernel invocation -------------------------------------------------

func(cuda.InOut(a), block=(4, 4, 1))
print "doubled with InOut:"
print a

# part 2 ----------------------------------------------------------------------

import pycuda.gpuarray as gpuarray
a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print "original array:"
print a_gpu
print "doubled with gpuarray:"
print a_doubled