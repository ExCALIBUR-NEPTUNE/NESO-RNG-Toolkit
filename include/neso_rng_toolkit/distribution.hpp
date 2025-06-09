#ifndef _NESO_RNG_TOOLKIT_DISTRIBUTION_HPP_
#define _NESO_RNG_TOOLKIT_DISTRIBUTION_HPP_
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace NESO::RNGToolkit {

namespace Distribution {

/**
 * For a value a find b such that a < b and the distance between a and b is as
 * small as possible for the type. This is a helper wrapper around
 * std::nextafter.
 *
 * @param value Value a to find next value from.
 * @returns Next value.
 */
template <typename VALUE_TYPE>
inline VALUE_TYPE next_value(const VALUE_TYPE a) {
  return std::nextafter(a, std::numeric_limits<VALUE_TYPE>::max());
}

/**
 * For a value a find b such that a < b and the distance between a and b is as
 * small as possible for the type. This is a helper wrapper around
 * std::nextafter. Avoids subnormals.
 *
 * @param value Value a to find next value from.
 * @returns Next value.
 */
inline double next_value(const double a) {
  constexpr double smallest_normal_pos = 2.2250738585072014e-308;
  constexpr double smallest_normal_neg = -2.2250738585072014e-308;
  const double next = std::nextafter(a, std::numeric_limits<double>::max());
  // Avoid sub-normals
  if ((smallest_normal_neg < next) && (next < smallest_normal_pos)) {
    constexpr double candidates[3] = {smallest_normal_neg, 0.0,
                                      smallest_normal_pos};
    int index = 0;
    double value;
    do {
      value = candidates[index++];
    } while (value <= a);
    return value;
  } else {
    return next;
  }
}

/**
 * For a value a find b such that a > b and the distance between a and b is as
 * small as possible for the type. This is a helper wrapper around
 * std::nextafter.
 *
 * @param value Value a to find next value from.
 * @returns Next value.
 */
template <typename VALUE_TYPE>
inline VALUE_TYPE previous_value(const VALUE_TYPE a) {
  return std::nextafter(a, std::numeric_limits<VALUE_TYPE>::lowest());
}

/**
 * For a value a find b such that a > b and the distance between a and b is as
 * small as possible for the type. This is a helper wrapper around
 * std::nextafter. Avoids subnormals.
 *
 * @param value Value a to find next value from.
 * @returns Next value.
 */
inline double previous_value(const double a) {
  constexpr double smallest_normal_pos = 2.2250738585072014e-308;
  constexpr double smallest_normal_neg = -2.2250738585072014e-308;
  const double next = std::nextafter(a, std::numeric_limits<double>::lowest());
  // Avoid sub-normals
  if ((smallest_normal_neg < next) && (next < smallest_normal_pos)) {
    constexpr double candidates[3] = {smallest_normal_pos, 0.0,
                                      smallest_normal_neg};
    int index = 0;
    double value;
    do {
      value = candidates[index++];
    } while (value >= a);
    return value;
  } else {
    return next;
  }
}

/**
 * Samples should be uniformly distributed in [a, b). Use the functions
 * next_value and previous_value to sample in (a, b), (a, b] and [a,b] as
 * required.
 */
template <typename VALUE_TYPE> struct Uniform {
  VALUE_TYPE a{0.0};
  VALUE_TYPE b{1.0};
};

/**
 * Samples should be distributed ~Normal(mean, stddev*stddev).
 */
template <typename VALUE_TYPE> struct Normal {
  VALUE_TYPE mean{0.0};
  VALUE_TYPE stddev{1.0};
};

} // namespace Distribution

} // namespace NESO::RNGToolkit

#endif
