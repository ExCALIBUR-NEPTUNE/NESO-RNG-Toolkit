#ifndef _NESO_RNG_TOOLKIT_PLATFORM_HPP_
#define _NESO_RNG_TOOLKIT_PLATFORM_HPP_

#include "distribution.hpp"
#include "rng.hpp"
#include "typedefs.hpp"
#include <set>

namespace NESO::RNGToolkit {

/**
 * This is the main interface to abstract the different platforms/vendors.
 */
template <typename VALUE_TYPE> struct Platform {
protected:
  /**
   * Helper function to check the passed generator is valid.
   *
   * @param generator_name Generator to check.
   * @param generators Set of acceptable generator names.
   * @returns True if generator in set.
   */
  static inline bool
  check_generator_name(const std::string generator_name,
                       const std::set<std::string> generators) {
    return generators.count(generator_name);
  }

  /**
   * @param generator_name generator_name.
   * @param default_generator_name.
   * @returns default_generator_name if generator_name is "default".
   */
  static inline std::string
  get_generator_name(std::string generator_name,
                     const std::string default_generator_name) {
    if (generator_name == "default") {
      return default_generator_name;
    } else {
      return generator_name;
    }
  }

public:
  virtual ~Platform() = default;

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @param generator_name Name of preferred RNG generator method.
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Uniform<VALUE_TYPE> distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index,
             std::string generator_name) = 0;

  /*
   * Create an RNG instance.
   *
   * @param distribution Distribution RNG samples should be from.
   * @param seed Value to seed RNG with.
   * @param device SYCL Device samples are to be created on.
   * @param device_index Index of SYCL device on the SYCL platform.
   * @param generator_name Name of preferred RNG generator method,
   * default="default".
   * @returns RNG instance. nullptr on Error.
   */
  virtual RNGSharedPtr<VALUE_TYPE>
  create_rng(Distribution::Normal<VALUE_TYPE> distribution, std::uint64_t seed,
             sycl::device device, std::size_t device_index,
             std::string generator_name) = 0;
};

} // namespace NESO::RNGToolkit

#endif
