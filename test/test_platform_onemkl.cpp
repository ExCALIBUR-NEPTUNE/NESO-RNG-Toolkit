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

  // Generate host side values to test against
  // TODO

  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));
  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N, 511) == SUCCESS);
  std::vector<VALUE_TYPE> to_test(N);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();

  for (auto ix : to_test) {
    std::cout << ix << std::endl;
  }

  sycl::free(d_ptr, queue);
}

} // namespace

TEST(PlatformOneMKL, uniform_double) { wrapper_uniform<double>(); }

#endif
