#ifdef NESO_RNG_TOOLKIT_ONEMKL
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

  const VALUE_TYPE a = -2.0;
  const VALUE_TYPE b = 2.0;

  auto to_test_rng =
      create_rng<VALUE_TYPE>(Distribution::Uniform<VALUE_TYPE>{a, b}, seed,
                             device, 0, "oneMKL", "default_engine");
  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));

  std::vector<VALUE_TYPE> correct(N);
  std::vector<VALUE_TYPE> to_test(N);

  oneapi::mkl::rng::default_engine engine(queue, seed);
  oneapi::mkl::rng::uniform<VALUE_TYPE> distr(a, b);

  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
  queue.fill(d_ptr, 0.0, N).wait_and_throw();
  oneapi::mkl::rng::generate(distr, engine, N, d_ptr).wait_and_throw();
  queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
  ASSERT_EQ(correct, to_test);

  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
  queue.fill(d_ptr, 0.0, N).wait_and_throw();

  bool one_different = false;
  for (std::size_t ix = 0; ix < N; ix++) {
    if (correct.at(ix) != to_test.at(ix)) {
      one_different = true;
    }
  }
  ASSERT_TRUE(one_different);

  oneapi::mkl::rng::generate(distr, engine, N, d_ptr).wait_and_throw();
  queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
  ASSERT_EQ(correct, to_test);

  sycl::free(d_ptr, queue);
}

template <typename VALUE_TYPE> inline void wrapper_normal() {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  const std::size_t N = 102300;
  const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

  const VALUE_TYPE mean = 3.0;
  const VALUE_TYPE stddev = 2.0;

  auto to_test_rng =
      create_rng<VALUE_TYPE>(Distribution::Normal<VALUE_TYPE>{mean, stddev},
                             seed, device, 0, "oneMKL", "default_engine");
  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));

  std::vector<VALUE_TYPE> correct(N);
  std::vector<VALUE_TYPE> to_test(N);

  oneapi::mkl::rng::default_engine engine(queue, seed);
  oneapi::mkl::rng::gaussian<VALUE_TYPE> distr(mean, stddev);

  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
  queue.fill(d_ptr, 0.0, N).wait_and_throw();
  oneapi::mkl::rng::generate(distr, engine, N, d_ptr).wait_and_throw();
  queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
  ASSERT_EQ(correct, to_test);

  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N) == SUCCESS);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
  queue.fill(d_ptr, 0.0, N).wait_and_throw();

  bool one_different = false;
  for (std::size_t ix = 0; ix < N; ix++) {
    if (correct.at(ix) != to_test.at(ix)) {
      one_different = true;
    }
  }
  ASSERT_TRUE(one_different);

  oneapi::mkl::rng::generate(distr, engine, N, d_ptr).wait_and_throw();
  queue.memcpy(correct.data(), d_ptr, num_bytes).wait_and_throw();
  ASSERT_EQ(correct, to_test);

  sycl::free(d_ptr, queue);
}

} // namespace

TEST(PlatformOneMKL, uniform_double) { wrapper_uniform<double>(); }
TEST(PlatformOneMKL, normal_double) { wrapper_normal<double>(); }
TEST(PlatformOneMKL, default) {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  const double mean = 3.0;
  const double stddev = 2.0;

  auto to_test_rng = create_rng<double>(
      Distribution::Normal<double>{mean, stddev}, seed, device, 0);

  ASSERT_EQ(to_test_rng->platform_name, "oneMKL");
}

#endif
