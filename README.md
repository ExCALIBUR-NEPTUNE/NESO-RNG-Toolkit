# NESO-RNG-Toolkit

NESO-RNG-Toolkit is an SYCL interface to vendor specific Random Number Generation (RNG) implementations.
We refer to these implementations as "platforms".
These are the currently supported platforms:

| Platform Name | Description |
| ------------- | ----------- |
| `stdlib`      | The C++ standard library random implementations. |
| `curand`      | CUDA provided RNG samples via cuRAND. |
| `oneMKL`      | Intel SYCL provided RNG samples via oneMKL. |

## Requirements

* SYCL 2020 implementation. e.g. oneAPI or AdaptiveCpp.
* (Optional) oneMKL.
* (Optional) cuRAND.

## CMake

The standard library RNG implementations, platform `stdlib`, are always made available. 
Vendor specific implementations are searched for at CMake time and are enabled if they are available.
The following table lists the options which can be passed to CMake to configure searching for vendor provided RNG implementations.

| CMake Variable Name | Default | Description |
| ------------------- | ------- | ----------- |
| `NESO_RNG_TOOLKIT_ENABLE_ONEMKL` | `ON` | Search for oneMKL in a non-fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_REQUIRE_ONEMKL` | `OFF` | Search for oneMKL in a fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_ENABLE_CURAND` | `ON` | Search for cuRAND in a non-fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_REQUIRE_CURAND` | `OFF` | Search for cuRAND in a fatal manner if it is not found. |


Downstream projects which use NESO-RNG-Toolkit should write CMake implementation that looks like the following example. 
Please see the examples directory for a NESO-Particles example.
```
find_package(NESO-RNG-Toolkit REQUIRED)
target_link_libraries(${EXECUTABLE} PUBLIC NESO-RNG-Toolkit::NESO-RNG-Toolkit)
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${EXECUTABLE_SOURCE})
```

## Distributions
Currently we support the following distributions for RNG samples. 
These interfaces should follow the C++ standard for definitions.

| Distribution Type | Description |
| ----------------- | ----------- |
| `Distribution::Uniform` | This describes a Uniform distribution over an interval [a, b). |
| `Distribution::Normal` | This describes a Normal distribution with mean `mean` and standard deviation `stddev`. |


Sometimes it is desirable to alter the interval in which uniform samples exist in to explicitly include or exclude the end points. We provide the functions `next_value` and `previous_value` to move the specification of the interval by one double precision value to include or exclude end points as needed.

| Interval Required | Uniform Distribution Call |
| ----------------- | ------------------------- |
| [a, b)            | `Distribution::Uniform{a, b}` |
| [a, b]            | `Distribution::Uniform{a, Distribution::next_value(b)}` |
| (a, b)            | `Distribution::Uniform{Distribution::next_value(a), b}` |
| (a, b]            | `Distribution::Uniform{Distribution::next_value(a), Distribution::next_value(b)}` |

## Interface

The main interface for this library is the `RNG` type which is described as follows.
```cpp
/**
 * This is the main interface for the library.
 */
template <typename VALUE_TYPE> struct RNG {

  /// The SYCL device this RNG is created on.
  sycl::device device;
  /// The index in the platform of the device.
  std::size_t device_index;
  /// The name of the platform.
  std::string platform_name{"undefined"};

  /**
   * Start to draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to fill with num_samples samples.
   * @param[in] num_samples Number of samples to place in device buffer.
   * @returns Error code to be tested against SUCCESS.
   */
  virtual int submit_get_samples(VALUE_TYPE *d_ptr,
                                 const std::size_t num_samples) = 0;

  /**
   * Start to draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to which is currently being populated
   * with samples.
   * @returns Error code to be tested against SUCCESS.
   */
  virtual int wait_get_samples(VALUE_TYPE *d_ptr) = 0;

  /**
   * Draw random samples from the RNG. Internally this function calls
   * submit_get_samples and wait_get_samples.
   *
   * @param[in, out] d_ptr Device pointer to fill with num_samples samples.
   * @param[in] num_samples Number of samples to place in device buffer.
   * @returns Error code to be tested against SUCCESS.
   */
  int get_samples(VALUE_TYPE *d_ptr, const std::size_t num_samples);
};
```

To create instances of this type users should call the function `create_rng` which has the following interface:
```cpp
/**
 * This is the function users could call to create a RNG instance.
 *
 * @param distribution Distribution RNG samples should be from.
 * @param seed Value to seed RNG with.
 * @param device SYCL Device samples are to be created on.
 * @param device_index Index of SYCL device on the SYCL platform.
 * @param platform_name Name of preferred RNG platform, default="default".
 * @param generator_name Name of preferred RNG generator method,
 * default="default".
 * @returns RNG instance. nullptr on Error.
 */
template <typename VALUE_TYPE, typename DISTRIBUTION_TYPE>
[[nodiscard]] RNGSharedPtr<VALUE_TYPE>
create_rng(DISTRIBUTION_TYPE distribution, std::uint64_t seed,
           sycl::device device, std::size_t device_index,
           std::string platform_name = "default",
           std::string generator_name = "default")
```

For example to create a Uniform distribution in the interval [a,b) or a Normal distribution with mean "mean" and standard deviation "sdtdev" users should call `create_rng` as follows:
```cpp

auto rng_uniform = NESO::RNGToolkit::create_rng<double>(
    NESO::RNGToolkit::Distribution::Uniform<REAL>{a, b}, 
    seed,
    device, 
    device_index
);

auto rng_normal = NESO::RNGToolkit::create_rng<double>(
    NESO::RNGToolkit::Distribution::Normal<REAL>{mean, stddev}, 
    seed,
    device, 
    device_index
);

```

In these code listings `device` is a `sycl::device` instance. 
`device_index` is the index of the SYCL device in the SYCL platform. 
`seed` is the RNG seed which the RNG generator will be initialised with. 

To facilitate the creation of unique seeds across multiple processes, e.g. MPI ranks, we provide the helper function `create_seeds` which can be called as follows:

```cpp
// Create a seed on each MPI rank.
std::uint64_t root_seed = 12341351;
std::uint64_t seed = NESO::RNGToolkit::create_seeds(size, rank, root_seed);
```

## Runtime Configuration

Users can configure which RNG platform and vendor specific RNG implementation is called at runtime through environment variables. 
We implement an interface to a restricted set of the RNG implementations provided by each vendor.
If you would like the interface to allow use of a particular RNG implementation please raise an issue.

| Variable Name | Description |
| ------------- | ----------- |
| `NESO_RNG_TOOLKIT_PLATFORM` | Explicitly specify which platform should provide the random samples. The acceptable values are the platform names in the first table. |
| `NESO_RNG_TOOLKIT_GENERATOR` | Explicitly specify which RNG generator provided by the vendor should be used. See the table below for acceptable values. |
| `NESO_RNG_TOOLKIT_PLATFORM_VERBOSE` | Print to stdout information on which RNG implementation is in use at runtime. |


| Platform Name | Implemented Generators |
| ------------- | ---------------------- |
| `stdlib`      | `mt19937_64`           |
| `oneMKL`      | `default_engine`       |
| `curand`      | `default` (alias for `CURAND_RNG_PSEUDO_DEFAULT`) |


