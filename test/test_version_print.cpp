#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>

using namespace NESO::RNGToolkit;

namespace {

template <typename VALUE_TYPE> inline void wrapper_uniform() {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  const std::size_t N = 1023;
  const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

  auto to_test_rng = create_rng<VALUE_TYPE>(Distribution::Uniform{}, seed,
                                            device, 0, "stdlib");

  // Generate host side values to test against
  auto rng = std::mt19937_64{seed};
  auto dist = std::uniform_real_distribution<VALUE_TYPE>(0.0, 1.0);
  std::vector<VALUE_TYPE> correct(N);
  std::generate(correct.begin(), correct.end(), [&]() { return dist(rng); });

  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));
  to_test_rng->get_samples(d_ptr, N, 511);
  std::vector<VALUE_TYPE> to_test(N);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();

  ASSERT_EQ(to_test, correct);
  sycl::free(d_ptr, queue);
}

template <typename VALUE_TYPE> inline void wrapper_normal() {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  const std::size_t N = 1023;
  const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

  auto to_test_rng =
      create_rng<VALUE_TYPE>(Distribution::Normal{}, seed, device, 0, "stdlib");

  // Generate host side values to test against
  auto rng = std::mt19937_64{seed};
  auto dist = std::normal_distribution<VALUE_TYPE>(0.0, 1.0);
  std::vector<VALUE_TYPE> correct(N);
  std::generate(correct.begin(), correct.end(), [&]() { return dist(rng); });

  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));
  to_test_rng->get_samples(d_ptr, N, 511);
  std::vector<VALUE_TYPE> to_test(N);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();

  ASSERT_EQ(to_test, correct);
  sycl::free(d_ptr, queue);
}

} // namespace

TEST(PlatformStdLib, uniform_double) { wrapper_uniform<double>(); }

TEST(PlatformStdLib, uniform_float) { wrapper_uniform<float>(); }

TEST(PlatformStdLib, normal_double) { wrapper_uniform<double>(); }

TEST(PlatformStdLib, normal_float) { wrapper_uniform<float>(); }
