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
  /// The name of the platform.
  std::string platform_name{"undefined"};

  /**
   * Start to draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to fill with num_samples samples.
   * @param[in] num_samples Number of samples to place in device buffer.
   * @param[in] block_size Specify a desired block size to draw samples in (may
   * not be used).
   * @returns Error code to be tested against SUCCESS.
   */
  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples,
                                 const std::size_t block_size) = 0;

  /**
   * Start to draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to which is currently being populated
   * with samples.
   * @returns Error code to be tested against SUCCESS.
   */
  virtual int wait_get_samples(VALUE_TYPE *d_ptr) = 0;

  /**
   * Draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to fill with num_samples samples.
   * @param[in] num_samples Number of samples to place in device buffer.
   * @param[in] block_size Specify a desired block size to draw samples in (may
   * not be used).
   * @returns Error code to be tested against SUCCESS.
   */
  int get_samples(VALUE_TYPE *d_ptr, const std::size_t num_samples,
                  const std::size_t block_size) {
    int err = SUCCESS;
    if ((err = this->submit_get_samples(d_ptr, num_samples, block_size)) !=
        SUCCESS) {
      return err;
    }
    return this->wait_get_samples(d_ptr);
  }
};

template <typename VALUE_TYPE>
using RNGSharedPtr = std::shared_ptr<RNG<VALUE_TYPE>>;

} // namespace NESO::RNGToolkit

#endif
