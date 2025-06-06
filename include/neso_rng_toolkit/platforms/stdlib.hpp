#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_STDLIB_HPP_

#include "../platform.hpp"
#include "../rng.hpp"
#include <random>

namespace NESO::RNGToolkit {

template <typename VALUE_TYPE, typename RNG_TYPE, typename DIST_TYPE>
struct StdLibRNG : public RNG<VALUE_TYPE> {
  virtual ~StdLibRNG() = default;

  sycl::queue queue;
  RNG_TYPE rng;
  DIST_TYPE dist;

  virtual int wait_get_samples([[maybe_unused]] VALUE_TYPE *d_ptr) override {
    return SUCCESS;
  }

  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples,
                                 const std::size_t block_size) override {

    auto d_ptr_start = d_ptr;

    // Create the random number in blocks and copy to device blockwise.
    std::vector<VALUE_TYPE> block0(block_size);
    std::vector<VALUE_TYPE> block1(block_size);

    VALUE_TYPE *ptr_tmp;
    VALUE_TYPE *ptr_current = block0.data();
    VALUE_TYPE *ptr_next = block1.data();
    std::size_t num_numbers_moved = 0;

    sycl::event e;
    while (num_numbers_moved < num_samples) {

      // Create a block of samples
      const std::size_t num_to_memcpy =
          std::min(static_cast<std::size_t>(block_size),
                   num_samples - num_numbers_moved);
      for (std::size_t ix = 0; ix < num_to_memcpy; ix++) {
        ptr_current[ix] = this->dist(this->rng);
      }

      // Wait until the previous block finished copying before starting this
      // copy
      e.wait_and_throw();
      e = this->queue.memcpy(d_ptr, ptr_current,
                             num_to_memcpy * sizeof(VALUE_TYPE));
      d_ptr += num_to_memcpy;
      num_numbers_moved += num_to_memcpy;

      // swap ptr_current and ptr_next such that the new samples are written
      // into ptr_next whilst ptr_current is being copied to the device.
      ptr_tmp = ptr_current;
      ptr_current = ptr_next;
      ptr_next = ptr_tmp;
    }
    e.wait_and_throw();

    if (num_numbers_moved != num_samples) {
      std::cout << "Failed to copy samples to device." << std::endl;
      return -1;
    }

    if (d_ptr != d_ptr_start + num_samples) {
      std::cout << "Failed to copy samples to device." << std::endl;
      return -2;
    }

    return SUCCESS;
  }

  StdLibRNG(sycl::queue queue, std::uint64_t seed, DIST_TYPE dist)
      : queue(queue), rng(RNG_TYPE{seed}), dist(dist) {
    this->platform_name = "stdlib";
  }
};

/**
 * This is the main interface to the C++ stdlib random implementations.
 */
template <typename VALUE_TYPE>
struct StdLibPlatform : public Platform<VALUE_TYPE> {

  virtual ~StdLibPlatform() = default;

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index) override {
    sycl::queue queue(device);
    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<StdLibRNG<VALUE_TYPE, std::mt19937_64,
                                   std::uniform_real_distribution<VALUE_TYPE>>>(
            queue, seed, std::uniform_real_distribution<VALUE_TYPE>(0.0, 1.0)));
  }

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index) override {
    sycl::queue queue(device);
    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<StdLibRNG<VALUE_TYPE, std::mt19937_64,
                                   std::normal_distribution<VALUE_TYPE>>>(
            queue, seed, std::normal_distribution<VALUE_TYPE>(0.0, 1.0)));
  }
};

} // namespace NESO::RNGToolkit

#endif
