#include<torch/torch.h>
#include<cmath>

using namespace std;

__global__ void get_num_neighbours_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> num_neighbours) {

	extern __shared__ float s[];

	int batch_num = coordinates.size(0);

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.y + threadIdx.y;

	int natoms = natom_counts[batchID];

	int num_neighbour_atoms_i = 0;

	if (iatom < natoms) {

		float rix = coordinates[batchID][iatom][0];
		float riy = coordinates[batchID][iatom][1];
		float riz = coordinates[batchID][iatom][2];

		//now loop through all atoms for batchID

		for (int tile = 0; tile < int(ceil(float(natoms) / float(blockDim.y))); tile++) {
			int jdx = tile * blockDim.y + threadIdx.y;

			if (jdx < natoms) {
				for (int k = 0; k < 3; k++) {
					s[k * blockDim.y + threadIdx.y] = coordinates[batchID][jdx][k];
				}
			} else {
				for (int k = 0; k < 3; k++) {
					s[k * blockDim.y + threadIdx.y] = HUGE_VALF;
				}
			}

			__syncthreads();

			for (int j = 0; j < min(natoms, blockDim.y); j++) {

				float rijx = rix - s[0 * blockDim.y + j];
				float rijy = riy - s[1 * blockDim.y + j];
				float rijz = riz - s[2 * blockDim.y + j];

				float rij2 = rijx * rijx + rijy * rijy + rijz * rijz;

				if (rij2 < rcut2 && rij2 > 0) {
					num_neighbour_atoms_i++;
				}
			}
		}

		num_neighbours[batchID][iatom] = num_neighbour_atoms_i;
	}
}

__global__ void get_neighbour_list_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts, float rcut2,
		torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbour_list) {

	extern __shared__ float s[];

	int batch_num = coordinates.size(0);

	int batchID = blockIdx.x;
	int iatom = blockIdx.y * blockDim.y + threadIdx.y;

	int natoms = natom_counts[batchID];
	int count = 0;

	if (iatom < natoms) {

		float rix = coordinates[batchID][iatom][0];
		float riy = coordinates[batchID][iatom][1];
		float riz = coordinates[batchID][iatom][2];

		//now loop through all atoms for batchID

		for (int tile = 0; tile < int(ceil(float(natoms) / float(blockDim.y))); tile++) {
			int jdx = tile * blockDim.y + threadIdx.y;

			if (jdx < natoms) {
				for (int k = 0; k < 3; k++) {
					s[k * blockDim.y + threadIdx.y] = coordinates[batchID][jdx][k];
				}
			} else {
				for (int k = 0; k < 3; k++) {
					s[k * blockDim.y + threadIdx.y] = HUGE_VALF;
				}
			}

			__syncthreads();

			for (int j = 0; j < min(natoms, blockDim.y); j++) {

				int jidx = tile * blockDim.y + j;

				float rijx = rix - s[0 * blockDim.y + j];
				float rijy = riy - s[1 * blockDim.y + j];
				float rijz = riz - s[2 * blockDim.y + j];

				float rij2 = rijx * rijx + rijy * rijy + rijz * rijz;

				if (rij2 < rcut2 && rij2 > 0) {
					neighbour_list[batchID][iatom][count] = jidx;
					count++;
				}
			}
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

	dim3 numBlocks(currBatchSize, int(ceil(float(natoms) / float(nthreads))));
	dim3 threadsPerBlock(1, nthreads);

safe_fill_kernel<<<numBlocks, threadsPerBlock>>>( pairlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>());

}

void getNumNeighboursCUDA(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor num_neighbours) {

int currBatchSize = coordinates.size(0);
int natoms = natom_counts.max().item<int>();

float rcut2 = rcut * rcut;

const int nthreads = 64;

dim3 numBlocks(currBatchSize, int(ceil(float(natoms) / float(nthreads))));
dim3 threadsPerBlock(1, nthreads);

get_num_neighbours_kernel<<<numBlocks, threadsPerBlock, 3 * nthreads * sizeof(float)>>>(
		coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		rcut2, num_neighbours.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

cudaDeviceSynchronize();

}

void getNeighbourListCUDA(torch::Tensor coordinates, torch::Tensor natom_counts, float rcut, torch::Tensor neighbour_list) {

int currBatchSize = coordinates.size(0);
int natoms = natom_counts.max().item<int>();

float rcut2 = rcut * rcut;

const int nthreads = 64;

dim3 numBlocks(currBatchSize, int(ceil(float(natoms) / float(nthreads))));
dim3 threadsPerBlock(1, nthreads);

get_neighbour_list_kernel<<<numBlocks, threadsPerBlock, 3 * nthreads * sizeof(float)>>>(
		coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
		natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
		rcut2,
		neighbour_list.packed_accessor32<int, 3, torch::RestrictPtrTraits>());

cudaDeviceSynchronize();
}

