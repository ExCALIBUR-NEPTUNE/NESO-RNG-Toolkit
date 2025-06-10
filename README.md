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


