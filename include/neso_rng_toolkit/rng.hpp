#ifndef _NESO_RNG_TOOLKIT_RNG_HPP_
#define _NESO_RNG_TOOLKIT_RNG_HPP_

#include "typedefs.hpp"

namespace NESO::RNGToolkit {

/**
 * This is the main interface for the library.
 */
template <typename VALUE_TYPE> struct RNG {

  /// The SYCL device this RNG is created on.
  sycl::device device;
  /// The index in the platform of the device.
  std::size_t device_index;

  /**
   * Draw random samples from the RNG.
   *
   * @param[in, out] d_ptr Device pointer to fill with num_samples samples.
   * @param[in] num_samples Number of samples to place in device buffer.
   * @returns Error code to be tested against SUCCESS.
   */
  virtual int get_samples(VALUE_TYPE *d_ptr, const std::size_t num_samples) = 0;
};

template <typename VALUE_TYPE>
using RNGSharedPtr = std::shared_ptr<RNG<VALUE_TYPE>>;
} // namespace NESO::RNGToolkit

#endif
