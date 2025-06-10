#ifdef NESO_RNG_TOOLKIT_CURAND
#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>

using namespace NESO::RNGToolkit;

namespace {

template <typename VALUE_TYPE> inline void wrapper_uniform() {
  sycl::device device{sycl::default_selector_v};
  if (device.is_gpu()) {
    sycl::queue queue{device};

    const std::uint64_t seed = 1234;
    const std::size_t N = 10230;
    const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

    const VALUE_TYPE a = -2.0;
    const VALUE_TYPE b = 2.0;

    auto to_test_rng =
        create_rng<VALUE_TYPE>(Distribution::Uniform<VALUE_TYPE>{a, b}, seed,
                               device, 0, "curand", "default");

    ASSERT_EQ(to_test_rng->platform_name, "curand");
    VALUE_TYPE *d_ptr =
        static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));

    std::vector<VALUE_TYPE> correct(N);
    std::vector<VALUE_TYPE> to_test(N);

    ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
    queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();

    std::shared_ptr<CurandRNG<VALUE_TYPE>> cast_rng =
        std::dynamic_pointer_cast<CurandRNG<VALUE_TYPE>>(to_test_rng);
    ASSERT_NE(cast_rng, nullptr);

    curandGenerator_t generator;

    ASSERT_TRUE(
        check_error_code(curandCreateGenerator(&generator, cast_rng->rng)));
    ASSERT_TRUE(check_error_code(curandSetStream(generator, cast_rng->stream)));
    ASSERT_TRUE(
        check_error_code(curandSetPseudoRandomGeneratorSeed(generator, seed)));

    ASSERT_TRUE(
        check_error_code(curandGenerateUniformDouble(generator, d_ptr, N)));
    ASSERT_TRUE(check_error_code(cudaStreamSynchronize(cast_rng->stream)));

    auto lambda_transform = [&]() {
      const VALUE_TYPE k_a = a;
      const VALUE_TYPE k_b = b;
      const VALUE_TYPE k_width = k_b - k_a;
      const VALUE_TYPE k_max_allowed_value = Distribution::previous_value(b);

      queue
          .parallel_for(sycl::range<1>(N),
                        [=](auto idx) {
                          const VALUE_TYPE original = d_ptr[idx];
                          // Transform the interval from (0, 1] to [0, 1).
                          const VALUE_TYPE swapped_interval = 1.0 - original;
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

    lambda_transform();

    queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();
    ASSERT_EQ(correct, to_test);

    ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
    queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();

    ASSERT_TRUE(
        check_error_code(curandGenerateUniformDouble(generator, d_ptr, N)));
    ASSERT_TRUE(check_error_code(cudaStreamSynchronize(cast_rng->stream)));
    lambda_transform();

    std::vector<VALUE_TYPE> correct_prev = correct;
    queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();
    ASSERT_EQ(correct, to_test);

    bool one_different = false;
    for (std::size_t ix = 0; ix < N; ix++) {
      if (correct.at(ix) != correct_prev.at(ix)) {
        one_different = true;
      }
    }
    ASSERT_TRUE(one_different);

    ASSERT_TRUE(check_error_code(curandDestroyGenerator(generator)));
    sycl::free(d_ptr, queue);
  }
}

template <typename VALUE_TYPE> inline void wrapper_normal() {
  sycl::device device{sycl::default_selector_v};
  if (device.is_gpu()) {
    sycl::queue queue{device};

    const std::uint64_t seed = 1234;
    const std::size_t N = 10230;
    const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

    const VALUE_TYPE mean = 2.0;
    const VALUE_TYPE stddev = 4.0;

    auto to_test_rng =
        create_rng<VALUE_TYPE>(Distribution::Normal<VALUE_TYPE>{mean, stddev},
                               seed, device, 0, "curand", "default");

    ASSERT_EQ(to_test_rng->platform_name, "curand");
    VALUE_TYPE *d_ptr =
        static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));

    std::vector<VALUE_TYPE> correct(N);
    std::vector<VALUE_TYPE> to_test(N);

    ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
    queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();

    std::shared_ptr<CurandRNG<VALUE_TYPE>> cast_rng =
        std::dynamic_pointer_cast<CurandRNG<VALUE_TYPE>>(to_test_rng);
    ASSERT_NE(cast_rng, nullptr);

    curandGenerator_t generator;

    ASSERT_TRUE(
        check_error_code(curandCreateGenerator(&generator, cast_rng->rng)));
    ASSERT_TRUE(check_error_code(curandSetStream(generator, cast_rng->stream)));
    ASSERT_TRUE(
        check_error_code(curandSetPseudoRandomGeneratorSeed(generator, seed)));

    ASSERT_TRUE(check_error_code(
        curandGenerateNormalDouble(generator, d_ptr, N, mean, stddev)));
    ASSERT_TRUE(check_error_code(cudaStreamSynchronize(cast_rng->stream)));

    queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();
    ASSERT_EQ(correct, to_test);

    ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
    queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();

    ASSERT_TRUE(check_error_code(
        curandGenerateNormalDouble(generator, d_ptr, N, mean, stddev)));
    ASSERT_TRUE(check_error_code(cudaStreamSynchronize(cast_rng->stream)));

    std::vector<VALUE_TYPE> correct_prev = correct;
    queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
    queue.fill(d_ptr, 0.0, N).wait_and_throw();
    ASSERT_EQ(correct, to_test);

    bool one_different = false;
    for (std::size_t ix = 0; ix < N; ix++) {
      if (correct.at(ix) != correct_prev.at(ix)) {
        one_different = true;
      }
    }
    ASSERT_TRUE(one_different);

    ASSERT_TRUE(check_error_code(curandDestroyGenerator(generator)));
    sycl::free(d_ptr, queue);
  }
}

} // namespace

TEST(PlatformCurand, uniform_double) { wrapper_uniform<double>(); }
TEST(PlatformCurand, normal_double) { wrapper_normal<double>(); }
TEST(PlatformCurand, uniform_double_host) {
  sycl::device device{sycl::cpu_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  auto to_test_rng = create_rng<double>(Distribution::Uniform<double>{0.0, 1.0},
                                        seed, device, 0, "curand", "default");

  // This cpu device should not create a curand RNG
  ASSERT_EQ(to_test_rng->platform_name, "stdlib");
}

#endif
