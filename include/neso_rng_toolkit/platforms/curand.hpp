#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_

#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"
#include <functional>
#include <iostream>

#ifdef NESO_RNG_TOOLKIT_CURAND
#include <cuda_runtime.h>
#include <curand.h>
#endif

namespace NESO::RNGToolkit {
#ifdef NESO_RNG_TOOLKIT_CURAND

inline bool check_error_code(curandStatus_t err) {
  if (err != CURAND_STATUS_SUCCESS) {
    std::cout << err << " = curandStatus_t != CURAND_STATUS_SUCCESS"
              << std::endl;
    return false;
  }
  return true;
}

inline bool check_error_code(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cout << err << " = cudaError_t != cudaSuccess" << std::endl;
    return false;
  }
  return true;
}

template <typename VALUE_TYPE> struct CurandRNG : RNG<VALUE_TYPE> {

  virtual ~CurandRNG() {
    check_error_code(curandDestroyGenerator(this->generator));
    check_error_code(cudaStreamDestroy(stream));
  }

  bool rng_good{true};
  cudaStream_t stream;
  std::size_t device_index;
  curandRngType_t rng;
  curandGenerator_t generator;

  std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *, std::size_t)>
      dist;

  virtual int wait_get_samples([[maybe_unused]] VALUE_TYPE *d_ptr) override {
    if (!this->rng_good) {
      return -1;
    }
    if (check_error_code(cudaStreamSynchronize(this->stream))) {
      return SUCCESS && this->rng_good;
    } else {
      return -1;
    }
  }

  virtual int
  submit_get_samples(VALUE_TYPE *d_ptr, const std::size_t num_samples,
                     [[maybe_unused]] const std::size_t block_size) override {
    if (!this->rng_good) {
      return -2;
    }
    if (check_error_code(this->dist(this->generator, d_ptr, num_samples))) {
      return SUCCESS && this->rng_good;
    } else {
      return -1;
    }
  }

  CurandRNG(std::size_t device_index, curandRngType_t rng, std::uint64_t seed,
            std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                         std::size_t)>
                dist)
      : device_index(device_index), rng(rng), dist(dist) {
    this->platform_name = "curand";

    this->rng_good =
        this->rng_good &&
        check_error_code(cudaSetDevice(static_cast<int>(device_index)));
    this->rng_good =
        this->rng_good && check_error_code(cudaStreamCreate(&this->stream));

    this->rng_good =
        this->rng_good &&
        check_error_code(curandCreateGenerator(&this->generator, this->rng));

    this->rng_good =
        this->rng_good &&
        check_error_code(curandSetStream(this->generator, this->stream));

    this->rng_good =
        this->rng_good && check_error_code(curandSetPseudoRandomGeneratorSeed(
                              this->generator, seed));
  }
};

template <typename VALUE_TYPE>
inline std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                    std::size_t)>
get_curand_uniform_dist(VALUE_TYPE &) {
  return [=](curandGenerator_t, VALUE_TYPE *, std::size_t) -> curandStatus_t {
    return CURAND_STATUS_TYPE_ERROR;
  };
}

inline std::function<curandStatus_t(curandGenerator_t, double *, std::size_t)>
get_curand_uniform_dist(double) {
  return [=](curandGenerator_t generator, double *d_ptr,
             std::size_t num_samples) -> curandStatus_t {
    return curandGenerateUniformDouble(generator, d_ptr, num_samples);
  };
}

inline std::function<curandStatus_t(curandGenerator_t, float *, std::size_t)>
get_curand_uniform_dist(float) {
  return [=](curandGenerator_t generator, float *d_ptr,
             std::size_t num_samples) -> curandStatus_t {
    return curandGenerateUniform(generator, d_ptr, num_samples);
  };
}

/**
 * This is the main interface to the curand random implementations.
 */
template <typename VALUE_TYPE>
struct CurandPlatform : public Platform<VALUE_TYPE> {

  const static inline std::set<std::string> generators = {"default"};

  virtual ~CurandPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, [[maybe_unused]] sycl::device device,
             std::size_t device_index, std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default");
    if (this->check_generator_name(generator_name, this->generators)) {

      std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                   std::size_t)>
          dist = get_curand_uniform_dist(static_cast<VALUE_TYPE>(0.0));

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<CurandRNG<VALUE_TYPE>>(
              device_index, CURAND_RNG_PSEUDO_DEFAULT, seed, dist));
      ;
    } else {
      return nullptr;
    }
  }

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
             std::uint64_t seed, [[maybe_unused]] sycl::device device,
             std::size_t device_index, std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default");
    if (this->check_generator_name(generator_name, this->generators)) {
      return nullptr;
    } else {
      return nullptr;
    }
  }
};

#else

/**
 * If curand is not found then we make the CurandPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using CurandPlatform = StdLibPlatform<VALUE_TYPE>;

#endif
} // namespace NESO::RNGToolkit

#endif
