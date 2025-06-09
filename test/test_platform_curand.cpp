#ifdef NESO_RNG_TOOLKIT_CURAND
#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>

using namespace NESO::RNGToolkit;

namespace {

template <typename VALUE_TYPE> inline void wrapper_uniform() {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};

  const std::uint64_t seed = 1234;
  const std::size_t N = 10230;
  const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

  const VALUE_TYPE a = -2.0;
  const VALUE_TYPE b = 2.0;

  auto to_test_rng =
      create_rng<VALUE_TYPE>(Distribution::Uniform<VALUE_TYPE>{a, b}, seed,
                             device, 0, "curand", "default");
  VALUE_TYPE *d_ptr =
      static_cast<VALUE_TYPE *>(sycl::malloc_device(num_bytes, queue));

  std::vector<VALUE_TYPE> correct(N);
  std::vector<VALUE_TYPE> to_test(N);

  ASSERT_TRUE(to_test_rng->get_samples(d_ptr, N, 511) == SUCCESS);
  queue.memcpy(to_test.data(), d_ptr, num_bytes).wait_and_throw();
  queue.fill(d_ptr, 0.0, N).wait_and_throw();

  sycl::free(d_ptr, queue);
}

} // namespace

TEST(PlatformCurand, uniform_double) { wrapper_uniform<double>(); }

#endif
