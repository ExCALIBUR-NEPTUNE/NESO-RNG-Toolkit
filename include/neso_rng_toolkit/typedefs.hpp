#ifndef _NESO_RNG_TOOLKIT_TYPEDEFS_HPP_
#define _NESO_RNG_TOOLKIT_TYPEDEFS_HPP_

#include <cstdint>
#include <sycl/sycl.hpp>

namespace NESO::RNGToolkit {

constexpr int SUCCESS = 0;

/**
 * Helper function to retrive string values from environment variables.
 *
 * @param key Name of environment variable.
 * @param default_value Default value to return if the key is not found.
 * @returns Value from environment variable.
 */
inline std::string get_env_string(const std::string key,
                                  std::string default_value) {
  char *var_char;
  const bool var_exists = (var_char = std::getenv(key.c_str())) != nullptr;
  if (var_exists) {
    try {
      std::string value = var_char;
      return value;
    } catch (...) {
      std::cout << "Could not read " + key +
             " and convert to int. Value of environment variable is: " + var_char +
             " Will return the default value of: " + default_value << std::endl;;
      return default_value;
    }
  } else {
    return default_value;
  }
}

}

#endif
