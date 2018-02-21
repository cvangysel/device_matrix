The `device_matrix` library
===========================

`device_matrix` is a lightweight, transparent, object-oriented and templated C++ library that encapsulates CUDA memory objects (i.e., tensors) and defines common operations on them.

Requirements & installation
---------------------------

To build the library and manage dependencies, we use [CMake](https://cmake.org/) (version 3.5 and higher). In addition, we rely on the following libraries:

   * [CUDA](https://developer.nvidia.com/cuda-zone) (version 8 and higher preferred),
   * [glog](https://github.com/google/glog) (version 0.3.4 and higher), and
   * [cnmem](https://github.com/NVIDIA/cnmem).

The tests are implemented using the [googletest and googlemock](https://github.com/google/googletest) frameworks, which CMake will fetch and compile automatically as part of the build pipeline. Finally, you need a CUDA-compatible GPU in order to perfrom any computations.

To install `device_matrix`, the following instructions should get you started.

	git clone https://github.com/cvangysel/device_matrix
	cd device_matrix
	mkdir build
	cd build
	cmake ..
	make
	make test
	make install
	
Please refer to the [CMake documentation](https://cmake.org/documentation) for advanced options.

Examples
--------

The following examples can also be found in the [examples](examples/) sub-directory of this repository. These examples will also be compiled as part of the build process.

### Matrix multiplication

``` cpp
#include <device_matrix/device_matrix.h>
	
#include <glog/logging.h>
#include <memory>

using namespace cuda;
	
int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
	
    const cudaStream_t stream = 0; // default CUDA stream.
	
    std::unique_ptr<device_matrix<float32>> a(
        device_matrix<float32>::create(
            stream,
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            2 /* num_rows */, 3 /* num_columns */));
	
    std::unique_ptr<device_matrix<float32>> b(
        device_matrix<float32>::create(
            stream,
            {7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
            3 /* num_rows */, 2 /* num_columns */));
	
    device_matrix<float32> c(
        2 /* num_rows */, 2 /* num_columns */, stream);
	
    matrix_mult(stream,
                *a, CUBLAS_OP_N,
                *b, CUBLAS_OP_N,
                &c);
	
    cudaDeviceSynchronize();
	
    print_matrix(c);
}
```
	
### Custom CUDA kernels

``` cpp
#include <device_matrix/device_matrix.h>

#include <glog/logging.h>
#include <memory>

using namespace cuda;

template <typename FloatT>
__global__
void inverse_kernel(FloatT* const input) {
    size_t offset = threadIdx.y * blockDim.x + threadIdx.x;
    input[offset] = -input[offset];
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    const cudaStream_t stream = 0; // default CUDA stream.

    std::unique_ptr<device_matrix<float32>> a(
        device_matrix<float32>::create(
            stream,
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            2 /* num_rows */, 3 /* num_columns */));

    LAUNCH_KERNEL(
        inverse_kernel
            <<<1, /* a single block */
               dim3(a->getRows(), a->getCols()), /* one thread per component */
               0,
               stream>>>(
            a->getData()));

    cudaDeviceSynchronize();

    print_matrix(*a);
}
```

Design principles
-----------------

`device_matrix` was explicitly designed to be inflexible with regards to variable passing/assignment as the lifetime of a `device_matrix` instance directly corresponds to the lifetime of the CUDA memory region it has allocated. That means that CUDA memory remains allocated as long as its underlying `device_matrix` exists and that `device_matrix` instances can only be passed as pointers or references. This gives total control of the CUDA memory allocation to the programmer, as it avoids garbage collection (e.g., Torch) or reference counting (e.g., `shared_ptr`), and allows for optimized CUDA memory usage. It uses [cnmem](https://github.com/NVIDIA/cnmem) for its memory management in order to avoid performance issues that occur due to the recurrent re-allocation of memory blocks of a particular size.

To avoid the implicit allocation of on-device memory, any operation resulting in a new allocation needs to be explicit in this. Most operations that return a new result will therefore reuse one of its inputs as destination memory space (in the process, the original input values will be overwritten!). As a result of this, C++ operators that imply value modification were deliberately omitted. 

The underlying CUDA memory space can easily be accessed by the library user. This allows the user to write arbitrary CUDA kernels that perform non-standard operations on CUDA objects in-place.

License
-------

`device_matrix` is licensed under the [MIT license](LICENSE). CUDA is a licensed trademark of NVIDIA. Please note that [CUDA](https://developer.nvidia.com/cuda-zone) is licensed separately.

If you modify `device_matrix` in any way, please link back to this repository.