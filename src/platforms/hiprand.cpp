#ifdef NESO_RNG_TOOLKIT_HIPRAND

#include <neso_rng_toolkit/platforms/hiprand.hpp>
#include <neso_rng_toolkit/platforms/hiprand_impl.hpp>

namespace NESO::RNGToolkit {

bool is_hip_device(sycl::device device, const std::size_t device_index) {}

template struct hipRANDPlatform<double>;
template struct hipRANDPlatform<float>;
} // namespace NESO::RNGToolkit
#endif
