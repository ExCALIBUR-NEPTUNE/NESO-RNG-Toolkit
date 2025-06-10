#include <neso_particles.hpp>
#include <neso_rng_toolkit.hpp>

using namespace NESO::Particles;

int main(int argc, char **argv) {
  initialise_mpi(&argc, &argv);

  const int npart_per_cell = 200;

  // Extent of each coarse cell in each dimension.
  const double cell_extent = 1.0;
  // Number of times to subdivide each coarse cell to create the mesh.
  const int subdivision_order = 0;
  const REAL fine_cell_extent = cell_extent / std::pow(2.0, subdivision_order);
  const int ndim = 2;
  std::vector<int> dims(ndim);
  // Number of coarse cells in the mesh in each dimension.
  const int Ncells = 8;
  dims[0] = Ncells;
  dims[1] = Ncells;

  // Halo width for local move.
  // Create the mesh.
  auto mesh = std::make_shared<CartesianHMesh>(
      MPI_COMM_WORLD, ndim, dims, cell_extent, subdivision_order, 1);

  // Create a container that wraps a sycl queue and a MPI communicator.
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  // Object to map particle positions into cells.
  auto cart_local_mapper = CartesianHMeshLocalMapper(sycl_target, mesh);
  // Create a domain from a mesh and a rule to map local cells to owning mpi
  // ranks.
  auto domain = std::make_shared<Domain>(mesh, cart_local_mapper);

  // The specification of the particle properties.
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("V"), ndim),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  // Create a ParticleGroup from a domain, particle specification and a compute
  // target.
  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  // Add some particles to each cell.
  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;
  std::mt19937 rng_pos(52234234 + rank);

  std::vector<std::vector<double>> positions;
  std::vector<int> cells;
  // Sample particles randomly in each local cell.
  uniform_within_cartesian_cells(mesh, npart_per_cell, positions, cells,
                                 rng_pos);

  // Host space to store the created particles.
  ParticleSet initial_distribution(cells.size(), A->get_particle_spec());
  // Populate the host space with particle data.
  for (int px = 0; px < cells.size(); px++) {
    for (int dimx = 0; dimx < ndim; dimx++) {
      initial_distribution[Sym<REAL>("P")][px][dimx] =
          positions.at(dimx).at(px);
    }
    initial_distribution[Sym<INT>("CELL_ID")][px][0] = cells.at(px);
    initial_distribution[Sym<INT>("ID")][px][0] = px;
  }
  // Add the particles to the ParticleGroup
  A->add_particles_local(initial_distribution);

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

  // Compute the mean and variance
  auto ga = std::make_shared<GlobalArray<REAL>>(sycl_target, 1);

  // Mean
  ga->fill(0.0);
  particle_loop(
      A,
      [=](auto GA, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          const auto v = V.at(dx);
          GA.add(0, v);
        }
      },
      Access::add(ga), Access::read(Sym<REAL>("V")))
      ->execute();
  const REAL npart_global_double = get_npart_global(A);
  const REAL mean = ga->get().at(0) / (npart_global_double * ndim);

  // Variance
  ga->fill(0.0);
  particle_loop(
      A,
      [=](auto GA, auto V) {
        for (int dx = 0; dx < ndim; dx++) {
          const auto v = V.at(dx) - mean;
          GA.add(0, v * v);
        }
      },
      Access::add(ga), Access::read(Sym<REAL>("V")))
      ->execute();
  const REAL variance = ga->get().at(0) / (npart_global_double * ndim);
  const REAL stddev = std::sqrt(variance);

  if (rank == 0) {
    sycl_target->print_device_info();
    nprint("mean:", mean, "stddev:", stddev);
  }

  sycl_target->free();
  mesh->free();

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }
  return 0;
}
