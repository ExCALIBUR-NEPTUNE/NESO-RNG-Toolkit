#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_IMPL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_ONEMKL_IMPL_HPP_
#ifdef NESO_RNG_TOOLKIT_ONEMKL

#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "onemkl.hpp"
#include "stdlib.hpp"
#include <oneapi/mkl.hpp>

namespace NESO::RNGToolkit {

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
    if (num_samples == 0) {
      this->event = sycl::event{};
    } else {
      this->event = oneapi::mkl::rng::generate(dist, rng, num_samples, d_ptr);
    }

    return SUCCESS;
  }

  oneMKLRNG(sycl::queue queue, RNG_TYPE rng, DIST_TYPE dist)
      : queue(queue), rng(rng), dist(dist) {
    this->platform_name = "oneMKL";
  }
};

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> OneMKLPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
    std::uint64_t seed, sycl::device device,
    [[maybe_unused]] std::size_t device_index, std::string generator_name) {
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

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> OneMKLPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
    std::uint64_t seed, sycl::device device,
    [[maybe_unused]] std::size_t device_index, std::string generator_name) {
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

} // namespace NESO::RNGToolkit

#endif
#endif
