#ifdef NESO_RNG_TOOLKIT_CURAND

#include <neso_rng_toolkit/platforms/curand.hpp>
#include <neso_rng_toolkit/platforms/curand_impl.hpp>

namespace NESO::RNGToolkit {

bool is_cuda_device(sycl::device device, const std::size_t device_index) {
  if (!device.is_gpu()) {
    return false;
  }

  bool is_cuda_device_flag = false;
  sycl::queue queue(device);
  void *d_ptr = sycl::malloc_device(8, queue);
  if (d_ptr == nullptr) {
    return false;
  }

  cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, d_ptr) == cudaSuccess) {
    if (attributes.type == cudaMemoryTypeDevice) {
      if (attributes.device != device_index) {
        std::cout
            << "Warning: The output of cudaPointerAttributes reports that the "
               "sycl::device and device_index may not be the same."
            << std::endl;
      }
      is_cuda_device_flag = true;
    }
  }

  sycl::free(d_ptr, queue);
  return is_cuda_device_flag;
}

template struct CurandPlatform<double>;
template struct CurandPlatform<float>;
} // namespace NESO::RNGToolkit
#endif
