#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_HPP_

#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"

#ifdef NESO_RNG_TOOLKIT_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace NESO::RNGToolkit {

#ifdef NESO_RNG_TOOLKIT_ONEMKL

template <typename VALUE_TYPE, typename RNG_TYPE, typename DIST_TYPE>
struct oneMKLRNG : RNG<VALUE_TYPE> {
  virtual ~oneMKLRNG() = default;

  sycl::queue queue;
  RNG_TYPE rng;
  DIST_TYPE dist;

  sycl::event event;

  virtual int wait_get_samples([[maybe_unused]] VALUE_TYPE *d_ptr) override {
    this->event.wait_and_throw();
    return SUCCESS;
  }

  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples) override {

    this->event = oneapi::mkl::rng::generate(dist, rng, num_samples, d_ptr);

    return SUCCESS;
  }

  oneMKLRNG(sycl::queue queue, RNG_TYPE rng, DIST_TYPE dist)
      : queue(queue), rng(rng), dist(dist) {
    this->platform_name = "oneMKL";
  }
};

/**
 * This is the main interface to the oneMKL random implementations.
 */
template <typename VALUE_TYPE>
struct OneMKLPlatform : public Platform<VALUE_TYPE> {

  const static inline std::set<std::string> generators = {"default_engine"};

  virtual ~OneMKLPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default_engine");
    if (this->check_generator_name(generator_name, this->generators)) {
      sycl::queue queue(device);
      auto engine = oneapi::mkl::rng::default_engine{queue, seed};
      auto dist =
          oneapi::mkl::rng::uniform<VALUE_TYPE>(distribution.a, distribution.b);

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<
              oneMKLRNG<VALUE_TYPE, decltype(engine), decltype(dist)>>(
              queue, engine, dist));
    } else {
      return nullptr;
    }
  }

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default_engine");
    if (this->check_generator_name(generator_name, this->generators)) {
      sycl::queue queue(device);
      auto engine = oneapi::mkl::rng::default_engine(queue, seed);
      auto dist = oneapi::mkl::rng::gaussian<VALUE_TYPE>(distribution.mean,
                                                         distribution.stddev);

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<
              oneMKLRNG<VALUE_TYPE, decltype(engine), decltype(dist)>>(
              queue, engine, dist));
    } else {
      return nullptr;
    }
  }
};

#else

/**
 * If oneMKL is not found then we make the OneMKLPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using OneMKLPlatform = StdLibPlatform<VALUE_TYPE>;
#endif

} // namespace NESO::RNGToolkit

#endif
