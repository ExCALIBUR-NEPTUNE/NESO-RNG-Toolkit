#ifdef NESO_RNG_TOOLKIT_HIPRAND
#include <gtest/gtest.h>
#include <neso_rng_toolkit.hpp>
#include <neso_rng_toolkit/platforms/hiprand_impl.hpp>

using namespace NESO::RNGToolkit;

TEST(PlatformhipRAND, base) {
  sycl::device device{sycl::default_selector_v};
  sycl::queue queue{device};
  ASSERT_TRUE(check_error_code(static_cast<hipError_t>(hipSuccess)));
}

namespace {

template <typename VALUE_TYPE> inline void wrapper_uniform() {
  sycl::device device{sycl::default_selector_v};
  if (device.is_gpu()) {
    sycl::queue queue{device};

    for (std::size_t N : {0, 1, 2, 3, 127, 301, 10238, 10239}) {
      for (std::size_t alignment_offset = 0; alignment_offset < 8;
           alignment_offset++) {

        const std::uint64_t seed = 1234;
        const std::size_t num_bytes = N * sizeof(VALUE_TYPE);

        const VALUE_TYPE a = -2.0;
        const VALUE_TYPE b = 2.0;

        auto to_test_rng =
            create_rng<VALUE_TYPE>(Distribution::Uniform<VALUE_TYPE>{a, b},
                                   seed, device, 0, "hipRAND", "default");

        ASSERT_EQ(to_test_rng->platform_name, "hipRAND");
      }
    }
  }
}

} // namespace

TEST(PlatformhipRAND, uniform_double) { wrapper_uniform<double>(); }

#endif
