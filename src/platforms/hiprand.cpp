#ifdef NESO_RNG_TOOLKIT_HIPRAND

#include <neso_rng_toolkit/platforms/hiprand.hpp>
#include <neso_rng_toolkit/platforms/hiprand_impl.hpp>

namespace NESO::RNGToolkit {

bool is_hip_device(sycl::device device, const std::size_t device_index) {

  if (!device.is_gpu()) {
    return false;
  }

  bool is_hip_device_flag = false;
  sycl::queue queue(device);
  void *d_ptr = sycl::malloc_device(8, queue);
  if (d_ptr == nullptr) {
    return false;
  }

  hipPointerAttribute_t attributes;
  check_error_code(hipPointerGetAttributes(&attributes, d_ptr));
  if (attributes.type == hipMemoryTypeDevice) {
    if (attributes.device != device_index) {
      std::cout
          << "Warning: The output of hipPointerGetAttributes reports that the "
             "sycl::device and device_index may not be the same."
          << std::endl;
    }
    is_hip_device_flag = true;
  }

  sycl::free(d_ptr, queue);
  return is_hip_device_flag;
}

template struct hipRANDPlatform<double>;
template struct hipRANDPlatform<float>;
} // namespace NESO::RNGToolkit
#endif
