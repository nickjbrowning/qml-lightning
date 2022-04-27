#include<torch/torch.h>
#include<cmath>

using namespace std;

__device__ void get_pbc_dij(float *drij, float *cell_vectors, float *inv_cell_vectors) {

	/*
	 *   h := [a, b, c], a=(a1,a2,a3), ... (the matrix of box vectors)
	 r_ij := r_i - r_j                 (difference vector)

	 s_i = h^{-1} r_i
	 s_ij = s_i - s_j
	 s_ij <-- s_ij - NINT(s_ij)        (general minimum image convention)
	 r_ij = h s_ij
	 */
	float sij[3] = { 0.0, 0.0, 0.0 };

	for (int m = 0; m < 3; m++) {

		for (int k = 0; k < 3; k++) {

			sij[m] += inv_cell_vectors[m * 3 + k] * drij[k];
		}

		sij[m] = sij[m] - round(sij[m]);
	}

	for (int m = 0; m < 3; m++) {

		float rij_m = 0.0;

		for (int k = 0; k < 3; k++) {
			rij_m += cell_vectors[m * 3 + k] * sij[k];
		}

		drij[m] = rij_m;
	}
}

__global__ void get_num_neighbours_kernel_shared(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lattice_vectors,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_lattice_vectors,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> num_neighbours) {

	extern __shared__ float s[];
	float *shared_x = (float*) &s;
	float *shared_y = (float*) &shared_x[blockDim.y];
	float *shared_z = (float*) &shared_y[blockDim.y];
	int *atomIDs = (int*) &shared_z[blockDim.y];
	float *slattice_vecs = (float*) &atomIDs[blockDim.y];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.y + threadIdx.y;

	int idy = threadIdx.y;

	int natoms = natom_counts[batchID];

	int ntiles = int(ceil(float(natoms) / blockDim.y));

	int num_neighbour_atoms_i = 0;

	float rix = -HUGE_VALF;
	float riy = -HUGE_VALF;
	float riz = -HUGE_VALF;

	bool pbc = false;

	float drij[3];

	if (lattice_vectors.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					slattice_vecs[i * 3 + j] = lattice_vectors[batchID][i][j];
					sinv_lattice_vecs[i * 3 + j] = inv_lattice_vectors[batchID][i][j];
				}
			}
		}
	}

	__syncthreads();

	if (iatom < natoms) {
		rix = coordinates[batchID][iatom][0];
		riy = coordinates[batchID][iatom][1];
		riz = coordinates[batchID][iatom][2];
	}

	__syncthreads();
//now loop through all atoms for batchID

	for (int tile = 0; tile < ntiles; tile++) {
		int jdx = tile * blockDim.y + threadIdx.y;

		if (jdx < natoms) {
			shared_x[idy] = coordinates[batchID][jdx][0];
			shared_y[idy] = coordinates[batchID][jdx][1];
			shared_z[idy] = coordinates[batchID][jdx][2];
			atomIDs[idy] = jdx;
		} else {
			shared_x[idy] = HUGE_VALF;
			shared_y[idy] = HUGE_VALF;
			shared_z[idy] = HUGE_VALF;
			atomIDs[idy] = -1;
		}

		__syncthreads();

		for (int j = 0; j < min(natoms, blockDim.y); j++) {

			if (atomIDs[j] == -1) {
				continue;
			}

			drij[0] = rix - shared_x[j];
			drij[1] = riy - shared_y[j];
			drij[2] = riz - shared_z[j];

			float rij2 = 0.0;

			if (pbc) {
				get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
			}

			rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];

			if (rij2 < rcut2 && rij2 > 0 && iatom < natoms) {
				num_neighbour_atoms_i++;
			}
		}
	}

	if (iatom < natoms) {
		num_neighbours[batchID][iatom] = num_neighbour_atoms_i;
	}
}

__global__ void get_num_neighbours_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lattice_vectors,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_lattice_vectors,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> num_neighbours) {

	extern __shared__ float s[];

	float *slattice_vecs = (float*) s;
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.x + threadIdx.x;

	int natoms = natom_counts[batchID];

	//printf("%i %i %i\n", batchID, iatom, natoms);

	int num_neighbour_atoms_i = 0;

	float rix = -HUGE_VALF;
	float riy = -HUGE_VALF;
	float riz = -HUGE_VALF;

	bool pbc = false;

	float drij[3];

	if (lattice_vectors.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					slattice_vecs[i * 3 + j] = lattice_vectors[batchID][i][j];
					sinv_lattice_vecs[i * 3 + j] = inv_lattice_vectors[batchID][i][j];
				}
			}
		}
	}

	__syncthreads();

	if (iatom < natoms) {
		rix = coordinates[batchID][iatom][0];
		riy = coordinates[batchID][iatom][1];
		riz = coordinates[batchID][iatom][2];
	} else {
		rix = -HUGE_VALF;
		riy = -HUGE_VALF;
		riz = -HUGE_VALF;
	}

	__syncthreads();
//now loop through all atoms for batchID

	for (int jdx = 0; jdx < natoms; jdx++) {

		drij[0] = rix - coordinates[batchID][jdx][0];
		drij[1] = riy - coordinates[batchID][jdx][1];
		drij[2] = riz - coordinates[batchID][jdx][2];

		float rij2 = 0.0;

		if (pbc) {
			get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
		}

		rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];

		if (rij2 < rcut2 && rij2 > 0 && iatom < natoms) {
			num_neighbour_atoms_i++;
		}
	}

	if (iatom < natoms) {
		num_neighbours[batchID][iatom] = num_neighbour_atoms_i;
	}
}

__global__ void get_neighbour_list_kernel_shared(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lattice_vectors,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_lattice_vectors,
		torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbour_list) {

	extern __shared__ float s[];
	float *shared_x = (float*) &s;
	float *shared_y = (float*) &shared_x[blockDim.y];
	float *shared_z = (float*) &shared_y[blockDim.y];
	int *atomIDs = (int*) &shared_z[blockDim.y];
	float *slattice_vecs = (float*) &atomIDs[blockDim.y];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.y + threadIdx.y;

	int idy = threadIdx.y;

	int natoms = natom_counts[batchID];
	int count = 0;

	int ntiles = int(ceil(float(natoms) / blockDim.y));

	float rix = -HUGE_VALF;
	float riy = -HUGE_VALF;
	float riz = -HUGE_VALF;

	bool pbc = false;

	float drij[3];

	if (lattice_vectors.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					slattice_vecs[i * 3 + j] = lattice_vectors[batchID][i][j];
					sinv_lattice_vecs[i * 3 + j] = inv_lattice_vectors[batchID][i][j];
				}
			}
		}
	}
	__syncthreads();

	if (iatom < natoms) {
		rix = coordinates[batchID][iatom][0];
		riy = coordinates[batchID][iatom][1];
		riz = coordinates[batchID][iatom][2];
	}

//now loop through all atoms for batchID

	for (int tile = 0; tile < ntiles; tile++) {
		int jdx = tile * blockDim.y + threadIdx.y;

		if (jdx < natoms) {
			shared_x[idy] = coordinates[batchID][jdx][0];
			shared_y[idy] = coordinates[batchID][jdx][1];
			shared_z[idy] = coordinates[batchID][jdx][2];
			atomIDs[idy] = jdx;
		} else {
			shared_x[idy] = HUGE_VALF;
			shared_y[idy] = HUGE_VALF;
			shared_z[idy] = HUGE_VALF;
			atomIDs[idy] = -1;
		}

		__syncthreads();

		for (int j = 0; j < min(natoms, blockDim.y); j++) {

			int jidx = atomIDs[j];

			if (jidx == -1) {
				continue;
			}

			drij[0] = rix - shared_x[j];
			drij[1] = riy - shared_y[j];
			drij[2] = riz - shared_z[j];

			float rij2 = 0.0;

			if (pbc) {
				get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
			}

			rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];

			if (rij2 < rcut2 && rij2 > 0 && iatom < natoms) {
				neighbour_list[batchID][iatom][count] = jidx;
				count++;
			}
		}
	}
}

__global__ void get_neighbour_list_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> lattice_vectors,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_lattice_vectors,
		torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbour_list) {

	extern __shared__ float s[];

	float *slattice_vecs = (float*) s;
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.x + threadIdx.x;

	int natoms = natom_counts[batchID];
	int count = 0;

	float rix = -HUGE_VALF;
	float riy = -HUGE_VALF;
	float riz = -HUGE_VALF;

	bool pbc = false;

	float drij[3];

	if (lattice_vectors.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					slattice_vecs[i * 3 + j] = lattice_vectors[batchID][i][j];
					sinv_lattice_vecs[i * 3 + j] = inv_lattice_vectors[batchID][i][j];
				}
			}
		}
	}
	__syncthreads();

	if (iatom < natoms) {
		rix = coordinates[batchID][iatom][0];
		riy = coordinates[batchID][iatom][1];
		riz = coordinates[batchID][iatom][2];
	}

	for (int jdx = 0; jdx < natoms; jdx++) {

		drij[0] = rix - coordinates[batchID][jdx][0];
		drij[1] = riy - coordinates[batchID][jdx][1];
		drij[2] = riz - coordinates[batchID][jdx][2];

		float rij2 = 0.0;

		if (pbc) {
			get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
		}

		rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];

		if (rij2 < rcut2 && rij2 > 0 && iatom < natoms) {
			neighbour_list[batchID][iatom][count] = jdx;
			count++;
		}
	}
}

__global__ void safe_fill_kernel(torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> pairlist) {

	int batch_num = pairlist.size(0);
	int natoms = pairlist.size(1);

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.y + threadIdx.y;

	if (iatom < natoms) {

		int newatm = -1;

		if (iatom == 0) {
			newatm = 1;
		} else {
			newatm = iatom - 1;
		}

		for (int k = 0; k < pairlist.size(2); k++) {
			int curval = pairlist[batchID][iatom][k];

			if (curval == -1) {
				pairlist[batchID][iatom][k] = newatm;
			}
		}
	}
}

void safeFillCUDA(torch::Tensor pairlist) {
	int currBatchSize = pairlist.size(0);
	int natoms = pairlist.size(1);

	const int nthreads = 64;

	dim3 numBlocks(currBatchSize, int(ceil(float(natoms) / nthreads)));
	dim3 threadsPerBlock(1, nthreads);

	safe_fill_kernel<<<numBlocks, threadsPerBlock>>>( pairlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void getNumNeighboursCUDA_shared(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor num_neighbours) {

	const int nthreads = 64;

	int currBatchSize = coordinates.size(0);
	int natoms = natom_counts.max().item<int>();

	int nBlockY = int(ceil(float(natoms) / nthreads));

	float rcut2 = rcut * rcut;

	dim3 numBlocks(currBatchSize, nBlockY);
	dim3 threadsPerBlock(1, nthreads);

	get_num_neighbours_kernel_shared<<<numBlocks, threadsPerBlock, (4 * nthreads + 18) * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			rcut2,
			lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			num_neighbours.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void getNumNeighboursCUDA(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor num_neighbours) {

	const int nthreads = 32;

	float rcut2 = rcut * rcut;

	dim3 numBlocks(coordinates.size(0), int(ceil((float) coordinates.size(1) / nthreads)));

	get_num_neighbours_kernel<<<numBlocks, nthreads, 18 * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			rcut2,
			lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			num_neighbours.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void getNeighbourListCUDA_shared(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor neighbour_list) {

	const int nthreads = 64;

	int currBatchSize = coordinates.size(0);
	int natoms = natom_counts.max().item<int>();

	int nBlockY = int(ceil(float(natoms) / nthreads));

	float rcut2 = rcut * rcut;

	dim3 numBlocks(currBatchSize, nBlockY);
	dim3 threadsPerBlock(1, nthreads);

	get_neighbour_list_kernel_shared<<<numBlocks, threadsPerBlock, (4 * nthreads + 18) * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			rcut2,
			lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			neighbour_list.packed_accessor32<int, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void getNeighbourListCUDA(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor neighbour_list) {

	const int nthreads = 32;

	dim3 numBlocks(coordinates.size(0), int(ceil((float) coordinates.size(1) / nthreads)));

	float rcut2 = rcut * rcut;

	get_neighbour_list_kernel<<<numBlocks, nthreads, 18 * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			rcut2,
			lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_lattice_vecs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			neighbour_list.packed_accessor32<int, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

