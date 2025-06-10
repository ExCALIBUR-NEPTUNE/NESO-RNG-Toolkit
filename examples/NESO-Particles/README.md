# Example With NESO-Particles

NESO-Particles (NP) is a NESO library for particle data and loop abstractions on unstructured meshes. 
As part of these loop abstractions users may want random numbers on a per particle basis.
NP provides a KernelRNG interface between the source of random numbers and the looping operations.
This example demonstrates how to use this interface with the RNG that NESO-RNG-Toolkit provides.

In this example we set particle velocities using samples from a Normal distribution with mean 4.0 and standard deviation 2.0.

```cpp
// Create a seed on each MPI rank.
std::uint64_t root_seed = 12341351;
std::uint64_t seed = NESO::RNGToolkit::create_seeds(size, rank, root_seed);

// Create a Normal distribution with mean 4.0 and standard deviation 2.0.
auto rng_normal = NESO::RNGToolkit::create_rng<REAL>(
    NESO::RNGToolkit::Distribution::Normal<REAL>{4.0, 2.0}, seed,
    sycl_target->device, sycl_target->device_index);

// Create an interface between NESO-RNG-Toolkit and NESO-Particles KernelRNG
auto rng_interface =
    make_rng_generation_function<GenericDeviceRNGGenerationFunction, REAL>(
        [=](REAL *d_ptr, const std::size_t num_samples) -> int {
          return rng_normal->get_samples(d_ptr, num_samples);
        });

// This is the object which can be passed to ParticleLoops and produces ndim
// samples per particle.
auto kernel_rng = host_atomic_block_kernel_rng<REAL>(rng_interface, ndim);

// Now use the rng in a kernel
particle_loop(
    A,
    [=](auto INDEX, auto RNG, auto V) {
      for (int dx = 0; dx < ndim; dx++) {
        bool valid;
        V.at(dx) = RNG.at(INDEX, dx, &valid);
      }
    },
    Access::read(ParticleLoopIndex{}), Access::read(kernel_rng),
    Access::write(Sym<REAL>("V")))
    ->execute();
```


