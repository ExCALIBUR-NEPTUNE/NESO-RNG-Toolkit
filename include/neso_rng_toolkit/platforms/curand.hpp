#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_HPP_

#include "../distribution.hpp"
#include "../platform.hpp"
#include "../platforms/stdlib.hpp"
#include "../rng.hpp"
#include "stdlib.hpp"
#include <functional>
#include <iostream>
#include <map>
#include <sycl/sycl.hpp>

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

/**
 * Try to determine if the SYCL device is actually a cuda device.
 *
 * @param device SYCL device.
 * @param device_index Device index in host sycl platform.
 * @returns True if this function determines that the device is a CUDA device.
 */
bool is_cuda_device(sycl::device device, const std::size_t device_index);

template <typename VALUE_TYPE> struct CurandRNG : RNG<VALUE_TYPE> {

  virtual ~CurandRNG() {
    check_error_code(curandDestroyGenerator(this->generator));
    check_error_code(cudaStreamDestroy(this->stream));
  }
  sycl::device device;
  sycl::queue queue;
  bool rng_good{true};
  cudaStream_t stream;
  std::size_t device_index;
  curandRngType_t rng;
  curandGenerator_t generator;

  std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *, std::size_t)>
      dist;

  std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform;

  std::map<VALUE_TYPE *, std::size_t> map_ptr_num_samples;

  virtual int wait_get_samples([[maybe_unused]] VALUE_TYPE *d_ptr) override {
    if (!this->rng_good) {
      return -1;
    }
    if (check_error_code(cudaStreamSynchronize(this->stream))) {
      const std::size_t num_samples = this->map_ptr_num_samples.at(d_ptr);
      if (num_samples > 0) {
        this->transform(this->queue, d_ptr, num_samples);
      }
      this->map_ptr_num_samples.erase(d_ptr);
      return this->rng_good ? SUCCESS : -3;
    } else {
      return -1;
    }
  }

  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples) override {
    if (!this->rng_good) {
      return -2;
    }

    this->map_ptr_num_samples[d_ptr] = num_samples;
    if (num_samples == 0) {
      return SUCCESS;
    }

    if (check_error_code(this->dist(this->generator, d_ptr, num_samples))) {
      return this->rng_good ? SUCCESS : -3;
    } else {
      return -1;
    }
  }

  CurandRNG(
      sycl::device device, std::size_t device_index, curandRngType_t rng,
      std::uint64_t seed,
      std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                   std::size_t)>
          dist,
      std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform)
      : device(device), queue(device), device_index(device_index), rng(rng),
        dist(dist), transform(transform) {
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

inline std::function<curandStatus_t(curandGenerator_t, double *, std::size_t)>
get_curand_normal_dist(const double mean, const double stddev) {
  return [=](curandGenerator_t generator, double *d_ptr,
             std::size_t num_samples) -> curandStatus_t {
    return curandGenerateNormalDouble(generator, d_ptr, num_samples, mean,
                                      stddev);
  };
}

inline std::function<curandStatus_t(curandGenerator_t, float *, std::size_t)>
get_curand_normal_dist(const float mean, const float stddev) {
  return [=](curandGenerator_t generator, float *d_ptr,
             std::size_t num_samples) -> curandStatus_t {
    return curandGenerateNormal(generator, d_ptr, num_samples, mean, stddev);
  };
}

/**
 * This is the main interface to the curand random implementations.
 */
template <typename VALUE_TYPE>
struct CurandPlatform : public Platform<VALUE_TYPE> {

  static const inline std::set<std::string> generators = {"default"};

  virtual ~CurandPlatform() = default;

  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng([[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
             std::uint64_t seed, sycl::device device, std::size_t device_index,
             std::string generator_name) override {
    generator_name = this->get_generator_name(generator_name, "default");
    if (this->check_generator_name(generator_name, this->generators)) {

      std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                   std::size_t)>
          dist = get_curand_uniform_dist(static_cast<VALUE_TYPE>(0.0));

      /**
       * Our interface follows the C++ standard and defines the interval as
       * [a,b). Curand samples values in (0,1]. Hence we transform the output to
       * be in [a,b).
       */

      sycl::queue queue{device};
      std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform =
          [=](sycl::queue queue, VALUE_TYPE *d_ptr, std::size_t num_samples) {
            const VALUE_TYPE k_max_allowed_value =
                Distribution::previous_value(distribution.b);
            const VALUE_TYPE k_a = distribution.a;
            const VALUE_TYPE k_b = distribution.b;
            const VALUE_TYPE k_width = k_b - k_a;
            queue
                .parallel_for(sycl::range<1>(num_samples),
                              [=](auto idx) {
                                const VALUE_TYPE original = d_ptr[idx];
                                // Transform the interval from (0, 1] to [0, 1).
                                const VALUE_TYPE swapped_interval =
                                    1.0 - original;
                                // Transform to [a, b);
                                VALUE_TYPE transform_interval =
                                    swapped_interval * k_width + k_a;
                                // Ensure after all that we are actually in [a,
                                // b)
                                transform_interval = (transform_interval < k_a)
                                                         ? k_a
                                                         : transform_interval;
                                transform_interval = (transform_interval >= k_b)
                                                         ? k_max_allowed_value
                                                         : transform_interval;
                                d_ptr[idx] = transform_interval;
                              })
                .wait_and_throw();
          };

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<CurandRNG<VALUE_TYPE>>(device, device_index,
                                                  CURAND_RNG_PSEUDO_DEFAULT,
                                                  seed, dist, transform));
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

      std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                   std::size_t)>
          dist = get_curand_normal_dist(distribution.mean, distribution.stddev);

      // No transform is needed for curand Normal distribution.
      sycl::queue queue{device};
      std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform =
          [=]([[maybe_unused]] sycl::queue queue,
              [[maybe_unused]] VALUE_TYPE *d_ptr,
              [[maybe_unused]] std::size_t num_samples) {};

      return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
          std::make_shared<CurandRNG<VALUE_TYPE>>(device, device_index,
                                                  CURAND_RNG_PSEUDO_DEFAULT,
                                                  seed, dist, transform));
      ;
    } else {
      return nullptr;
    }
  }
};

extern template struct CurandRNG<double>;
extern template struct CurandPlatform<double>;

#else

inline bool is_cuda_device(sycl::device, const std::size_t) { return false; }

/**
 * If curand is not found then we make the CurandPlatform a copy of the
 * StdLibPlatform.
 */
template <typename VALUE_TYPE>
using CurandPlatform = StdLibPlatform<VALUE_TYPE>;

#endif
} // namespace NESO::RNGToolkit

#endif
