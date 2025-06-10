# NESO-RNG-Toolkit

NESO-RNG-Toolkit is an SYCL interface to vendor specific Random Number Generation (RNG) implementations.
We refer to these implementations as "platforms".
These are the currently supported platforms:

| Platform Name | Description |
| ------------- | ----------- |
| `stdlib`      | The C++ standard library random implementations. |
| `curand`      | CUDA provided RNG samples via cuRAND. |
| `oneMKL`      | Intel SYCL provided RNG samples via oneMKL. |

The standard library RNG implementations, platform `stdlib`, are always made available. 
Vendor specific implementations are searched for at CMake time and are enabled if they are available.
The following table lists the options which can be passed to CMake to configure searching for vendor provided RNG implementations.

| CMake Variable Name | Default | Description |
| ------------------- | ------- | ----------- |
| `NESO_RNG_TOOLKIT_ENABLE_ONEMKL` | `ON` | Search for oneMKL in a non-fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_REQUIRE_ONEMKL` | `OFF` | Search for oneMKL in a fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_ENABLE_CURAND` | `ON` | Search for cuRAND in a non-fatal manner if it is not found. |
| `NESO_RNG_TOOLKIT_REQUIRE_CURAND` | `OFF` | Search for cuRAND in a fatal manner if it is not found. |


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


