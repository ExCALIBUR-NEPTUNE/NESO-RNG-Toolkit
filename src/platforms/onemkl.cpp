#ifdef NESO_RNG_TOOLKIT_ONEMKL

#include <neso_rng_toolkit/platforms/onemkl.hpp>
#include <neso_rng_toolkit/platforms/onemkl_impl.hpp>

namespace NESO::RNGToolkit {

template struct OneMKLPlatform<double>;
template struct OneMKLPlatform<float>;

} // namespace NESO::RNGToolkit

#endif
