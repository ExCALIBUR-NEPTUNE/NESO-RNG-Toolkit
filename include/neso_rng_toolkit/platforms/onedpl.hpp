#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_ONEDPL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_ONEDPL_HPP_

#include "../platform.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"
#include <functional>

#ifdef NESO_RNG_TOOLKIT_ONEDPL
#include <oneapi/dpl/random>
#include <oneapi/mkl.hpp>
#endif

namespace NESO::RNGToolkit {

#ifdef NESO_RNG_TOOLKIT_ONEDPL

template<typename VALUE_TYPE>
struct oneDPLRNG : RNG<VALUE_TYPE> {
  virtual ~oneDPLRNG() = default;

  sycl::event event;
  std::function<sycl::event(VALUE_TYPE *, const std::size_t)> submit_function;

  virtual int wait_get_samples([[maybe_unused]] VALUE_TYPE *d_ptr) override {
    this->event.wait_and_throw();
    return SUCCESS;
  }

  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples,
                                 [[maybe_unused]] const std::size_t block_size) override {
    this->event = this->submit_function(d_ptr, num_samples);
    return SUCCESS;
  }
  
  oneDPLRNG(
    std::function<sycl::event(VALUE_TYPE *, const std::size_t)> submit_function
  )
    : event(sycl::event{}), submit_function(submit_function)
  {
    this->platform_name = "oneDPL";
  }

};


/**
 * This is the main interface to the oneDPL random implementations.
 */
template <typename VALUE_TYPE>
struct OneDPLPlatform : public Platform<VALUE_TYPE> {

  const static inline std::set<std::string> generators = {"default_engine"};

  virtual ~OneDPLPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default_engine");
    if (this->check_generator_name(generator_name, this->generators)) {
      sycl::queue queue(device);
      std::function<sycl::event(VALUE_TYPE *, const std::size_t)>
          submit_function =
              [=](VALUE_TYPE *d_ptr, std::size_t num_samples) -> sycl::event {
        oneapi::mkl::rng::default_engine engine(queue, seed);
        // TODO CHECK LIMITS
        oneapi::mkl::rng::uniform<VALUE_TYPE> distr(distribution.a, distribution.b);
        auto event = oneapi::mkl::rng::generate(distr, engine, num_samples, d_ptr);
        return event;
      };

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<oneDPLRNG<VALUE_TYPE>>(submit_function));

    } else {
      return nullptr;
    }
  }


  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device,
             [[maybe_unused]] std::size_t device_index,
             std::string generator_name) override {
    sycl::queue queue(device);


  std::function<sycl::event(VALUE_TYPE *, const std::size_t)> submit_function = [=](
    VALUE_TYPE * d_ptr,
    std::size_t num_samples
  ) -> sycl::event {


    return sycl::event{};
  };

    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<oneDPLRNG<VALUE_TYPE>>(submit_function));
  }


};

#else
/**
 * If oneDPL is not found then we make the OneDPLPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using OneDPLPlatform = StdLibPlatform<VALUE_TYPE>;
#endif

} // namespace NESO::RNGToolkit

#endif
