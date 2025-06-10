#ifdef NESO_RNG_TOOLKIT_ONEMKL

#include <neso_rng_toolkit/platforms/onemkl.hpp>

namespace NESO::RNGToolkit {

template struct OneMKLPlatform<double>;
template struct OneMKLPlatform<float>;

} // namespace NESO::RNGToolkit

#endif
