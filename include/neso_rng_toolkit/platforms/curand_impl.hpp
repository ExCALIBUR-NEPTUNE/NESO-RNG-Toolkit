#ifndef _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_IMPL_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORMS_CURAND_IMPL_HPP_
#ifdef NESO_RNG_TOOLKIT_CURAND

#include "curand.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <type_traits>

namespace NESO::RNGToolkit {

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
    if (this->d_even_buffer != nullptr) {
      check_error_code(cudaFreeAsync(this->d_even_buffer, this->stream));
    }
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
  bool requires_even_number_of_samples{false};
  static const std::size_t even_buffer_size = 32;
  VALUE_TYPE *d_even_buffer{nullptr};

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

    // The cuRAND normal and lognormal (?) generators will error if the
    // alignment of the output buffers is not twice the standard alignment.
    const std::size_t offset_start =
        this->requires_even_number_of_samples
            ? (reinterpret_cast<std::uintptr_t>(d_ptr) %
               (std::alignment_of_v<VALUE_TYPE> * 2)) /
                  sizeof(VALUE_TYPE)
            : 0;

    // The cuRAND normal and lognormal generators will only sample an even
    // number of values and the pointers have to aligned to two values.
    std::size_t offset_end = 0;

    if (this->requires_even_number_of_samples) {
      // pointer is not aligned
      if (offset_start) {
        // offset for alignment by itself is not an even number of samples
        if ((num_samples - offset_start) % 2 != 0) {
          offset_end = 1;
        }
      } else {
        // No pointer offset is needed but an offset for odd number of samples
        // is need
        if ((num_samples) % 2 != 0) {
          offset_end = 1;
        }
      }
    }

    if (offset_end || offset_start) {
      // First get an even number of samples into the buffer.
      this->rng_good =
          this->rng_good &&
          check_error_code(this->dist(this->generator, this->d_even_buffer,
                                      this->even_buffer_size));
      this->rng_good = this->rng_good &&
                       check_error_code(cudaStreamSynchronize(this->stream));

      if (offset_start) {
        this->rng_good =
            this->rng_good &&
            check_error_code(cudaMemcpyAsync(
                d_ptr, this->d_even_buffer, offset_start * sizeof(VALUE_TYPE),
                cudaMemcpyDeviceToDevice, this->stream));
      }

      if (offset_end && ((offset_start + offset_end) <= num_samples)) {
        this->rng_good =
            this->rng_good &&
            check_error_code(cudaMemcpyAsync(
                d_ptr + num_samples - offset_end,
                this->d_even_buffer + this->even_buffer_size - offset_end,
                offset_end * sizeof(VALUE_TYPE), cudaMemcpyDeviceToDevice,
                this->stream));
      }

      this->rng_good = this->rng_good &&
                       check_error_code(cudaStreamSynchronize(this->stream));
    }

    if (!this->rng_good) {
      return -5;
    }

    // If we need any more samples
    if ((offset_start + offset_end) < num_samples) {

      const std::size_t num_samples_remaining =
          num_samples - offset_start - offset_end;

      if (this->requires_even_number_of_samples &&
          ((num_samples_remaining) % 2 == 1)) {
        std::cout
            << "Even number of samples required but number of samples is: " +
                   std::to_string(num_samples_remaining)
            << std::endl;
        return -6;
      }

      if (check_error_code(this->dist(this->generator, d_ptr + offset_start,
                                      num_samples_remaining))) {
        return this->rng_good ? SUCCESS : -3;
      } else {
        return -1;
      }
    } else {
      return this->rng_good ? SUCCESS : -4;
    }
  }

  CurandRNG(
      sycl::device device, std::size_t device_index, curandRngType_t rng,
      std::uint64_t seed,
      std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *,
                                   std::size_t)>
          dist,
      std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform,
      const bool requires_even_number_of_samples)
      : device(device), queue(device), device_index(device_index), rng(rng),
        dist(dist), transform(transform),
        requires_even_number_of_samples(requires_even_number_of_samples) {

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

    if (this->requires_even_number_of_samples) {
      this->rng_good =
          this->rng_good &&
          check_error_code(cudaMallocAsync(
              &(this->d_even_buffer),
              this->even_buffer_size * sizeof(VALUE_TYPE), this->stream));
      this->rng_good = this->rng_good &&
                       check_error_code(cudaStreamSynchronize(this->stream));
      this->rng_good = this->rng_good && (this->d_even_buffer != nullptr);
    }
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

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> CurandPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Uniform<VALUE_TYPE> distribution,
    std::uint64_t seed, sycl::device device, std::size_t device_index,
    std::string generator_name) {
  generator_name = this->get_generator_name(generator_name, "default");
  if (this->check_generator_name(generator_name, this->generators)) {

    std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *, std::size_t)>
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
                                                CURAND_RNG_PSEUDO_DEFAULT, seed,
                                                dist, transform, false));
    ;
  } else {
    return nullptr;
  }
}

template <typename VALUE_TYPE>
RNGSharedPtr<VALUE_TYPE> CurandPlatform<VALUE_TYPE>::create_rng(
    [[maybe_unused]] Distribution::Normal<VALUE_TYPE> distribution,
    std::uint64_t seed, [[maybe_unused]] sycl::device device,
    std::size_t device_index, std::string generator_name) {
  generator_name = this->get_generator_name(generator_name, "default");
  if (this->check_generator_name(generator_name, this->generators)) {

    std::function<curandStatus_t(curandGenerator_t, VALUE_TYPE *, std::size_t)>
        dist = get_curand_normal_dist(distribution.mean, distribution.stddev);

    // No transform is needed for curand Normal distribution.
    sycl::queue queue{device};
    std::function<void(sycl::queue, VALUE_TYPE *, std::size_t)> transform =
        [=]([[maybe_unused]] sycl::queue queue,
            [[maybe_unused]] VALUE_TYPE *d_ptr,
            [[maybe_unused]] std::size_t num_samples) {};

    return std::dynamic_pointer_cast<RNG<VALUE_TYPE>>(
        std::make_shared<CurandRNG<VALUE_TYPE>>(device, device_index,
                                                CURAND_RNG_PSEUDO_DEFAULT, seed,
                                                dist, transform, true));
    ;
  } else {
    return nullptr;
  }
}

} // namespace NESO::RNGToolkit
#endif
#endif
