#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

#define FULL_MASK 0xffffffff

__device__ void get_pbc_dij(float *drij, float *cell_vectors, float *inv_cell_vectors) {

	/*
	 *   h := [a, b, c], a=(a1,a2,a3), ... (the matrix of box vectors)
	 r_ij := r_i - r_j                 (difference vector)

	 s_i = h^{-1} r_i
	 s_ij = s_i - s_j
	 s_ij <-- s_ij - NINT(s_ij)        (general minimum image convention)
	 r_ij = h s_ij
	 */

	for (int x = 0; x < 3; x++) {

		float sij_x = 0.0;
		float rij_x = 0.0;

		for (int y = 0; y < 3; y++) {
			sij_x += inv_cell_vectors[x * 3 + y] * drij[x];
		}

		sij_x = sij_x - round(sij_x);

		for (int y = 0; y < 3; y++) {
			rij_x += cell_vectors[x * 3 + y] * sij_x;
		}

		drij[x] = rij_x;
	}
}

__global__ void egto_atomic_representation_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mbodylist,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gto_components,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> orbital_weights,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> gto_powers,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gridpoints,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> lchannel_weights,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> inv_factors, float eta, int lmax, float rcut,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int nbatch = coordinates.size(0);

	int ngauss = gridpoints.size(0);

	int norbs = gto_components.size(0);
	int nspecies = species.size(0);
	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	int krepsize = nmbody * norbs * ngauss;
	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *sgridpoints = (float*) &s; //ngaussians
	int *smbodylist = (int*) &sgridpoints[ngauss]; //nspecies * nspecies
	float *sgto_components_x = (float*) &smbodylist[nspecies * nspecies]; //norbs
	float *sgto_components_y = (float*) &sgto_components_x[norbs]; //norbs
	float *sgto_components_z = (float*) &sgto_components_y[norbs]; //norbs
	int *sgto_powers = (int*) &sgto_components_z[norbs]; //norbs
	float *sorbital_weights = (float*) &sgto_powers[norbs]; //norbs

	float *norb_temporary = (float*) &sorbital_weights[norbs]; //[norbs x nmbody x ngauss]
	float *lmax_temporary = (float*) &norb_temporary[nmbody * ngauss * norbs]; //[(lmax+1) x nmbody x ngauss]

	float *slattice_vecs = (float*) &lmax_temporary[(lmax + 1) * nmbody * ngauss];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];
	/*Shared Memory Definitions*/

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	float rix = coordinates[molID][iatom][0];
	float riy = coordinates[molID][iatom][1];
	float riz = coordinates[molID][iatom][2];

	double sqrt_eta = sqrt(eta / M_PI);

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x < 3) {
			for (int j = 0; j < 3; j++) {
				slattice_vecs[threadIdx.x * 3 + j] = cell[molID][threadIdx.x][j];
				sinv_lattice_vecs[threadIdx.x * 3 + j] = inv_cell[molID][threadIdx.x][j];
			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < norbs; i += blockDim.x) {
		sgto_components_x[i] = gto_components[i][0];
		sgto_components_y[i] = gto_components[i][1];
		sgto_components_z[i] = gto_components[i][2];

		sorbital_weights[i] = orbital_weights[i];

		sgto_powers[i] = gto_powers[i];
	}

	__syncthreads();

	for (int i = threadIdx.x; i < ngauss; i += blockDim.x) {
		sgridpoints[i] = gridpoints[i];
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int j = 0; j < nspecies; j++) {
			for (int k = j; k < nspecies; k++) {
				smbodylist[j * nspecies + k] = mbodylist[j][k];
				smbodylist[k * nspecies + j] = mbodylist[k][j];

			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < krepsize; i += blockDim.x) {
		norb_temporary[i] = 0.0;

	}

	__syncthreads();

	for (int i = threadIdx.x; i < lrepsize; i += blockDim.x) {
		lmax_temporary[i] = 0.0;
	}

	__syncthreads();

	float drij[3];

//save norb,ngaussian points into shared memory
	for (int jatom = 0; jatom < nneighbours_i; jatom++) {

		int j = neighbourlist[molID][iatom][jatom];

		if (j == -1) {
			continue;
		}

		float rjx = coordinates[molID][j][0];
		float rjy = coordinates[molID][j][1];
		float rjz = coordinates[molID][j][2];
		int element_type = element_types[molID][j];

//		if (element_type < 0 || element_type > nspecies - 1 && threadIdx.x == 1) {
//			printf("blockIdx.x: %d threadIdx.x: %d nneighbours: %d, neighbour: %d element_type: %d\n", blockIdx.x, threadIdx.x, nneighbours_i, j, element_type);
//
//			for (int c = 0; c < nneighbours_i; c++) {
//				int j = neighbourlist[molID][iatom][c];
//				printf("%d ", j);
//			}
//			printf("\n");
//
//		}

		__syncthreads();

		drij[0] = rix - rjx;
		drij[1] = riy - rjy;
		drij[2] = riz - rjz;

		if (pbc) {
			get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
		}

		float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

		float cut = 0.5 * (cos(rij * M_PI / rcut) + 1.0);

		for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

			int z = k % ngauss;
			int korb = int(floor(float(k) / ngauss));

			int gto_power = sgto_powers[korb];

			float gto_component_x = sgto_components_x[korb];
			float gto_component_y = sgto_components_y[korb];
			float gto_component_z = sgto_components_z[korb];

			float ang = pow(drij[0], gto_component_x) * pow(drij[1], gto_component_y) * pow(drij[2], gto_component_z);

			float val = sqrt_eta * (1.0 / pow(rij, inv_factors[gto_power] + gto_power)) * ang * cut;

			float gval = exp(-eta * pow(rij - sgridpoints[z], 2.0)) * val;

			for (int m = 0; m < nspecies; m++) {

				int ej = mbodylist[element_type][m];

				int kidx = korb * nmbody * ngauss + ej * ngauss + z;

				atomicAdd(&norb_temporary[kidx], gval);
			}
		}
	}

	__syncthreads();

//contract into (lmax+1) channels

//norb_temporary:[norbs, nmbody, ngauss]

	for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int korb = int(floor(float(k) / ngauss));

		int lchannel = sgto_powers[korb];

		for (int m = 0; m < nspecies; m++) {
			for (int n = m; n < nspecies; n++) {

				int mnidx = smbodylist[m * nspecies + n];

				int kmn = korb * nmbody * ngauss + mnidx * ngauss + z;
				int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

				float val = norb_temporary[kmn];

				//atomic needed here as multiple korb can hit the same L channel
				atomicAdd(&lmax_temporary[lmn], lchannel_weights[lchannel] * sorbital_weights[korb] * val * val);
			}
		}
	}
	__syncthreads();

//subtract single-element contributions
	for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int l = int(floor(float(k) / ngauss));

		for (int m = 0; m < nspecies; m++) {
			for (int n = m + 1; n < nspecies; n++) {

				int mnidx = smbodylist[m * nspecies + n];
				int mmidx = smbodylist[m * nspecies + m];
				int nnidx = smbodylist[n * nspecies + n];

				int lmn = l * nmbody * ngauss + mnidx * ngauss + z;
				int lmm = l * nmbody * ngauss + mmidx * ngauss + z;
				int lnn = l * nmbody * ngauss + nnidx * ngauss + z;

				float t1 = lmax_temporary[lmm];
				float t2 = lmax_temporary[lnn];

				atomicAdd(&lmax_temporary[lmn], -(t1 + t2));
				//lmax_temporary[lmn] -= (t1 + t2);
			}
		}
	}

	__syncthreads();

//save to global memory
	for (int k = threadIdx.x; k < lrepsize; k += blockDim.x) {
		output[molID][iatom][k] = lmax_temporary[k];
	}
}

__global__ void egto_atomic_representation_derivative_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mbodylist,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gto_components,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> orbital_weights,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> gto_powers,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gridpoints,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> lchannel_weights,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> inv_factors, float eta, int lmax, float rcut,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output, torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> grad) {

	extern __shared__ int s[];

	int nbatch = coordinates.size(0);
	int ngauss = gridpoints.size(0);
	int norbs = gto_components.size(0);
	int nspecies = species.size(0);
	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	int krepsize = nmbody * norbs * ngauss;
	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *sgridpoints = (float*) &s; //ngaussians
	int *smbodylist = (int*) &sgridpoints[ngauss]; //nspecies * nspecies
	float *sgto_components_x = (float*) &smbodylist[nspecies * nspecies]; //norbs
	float *sgto_components_y = (float*) &sgto_components_x[norbs]; //norbs
	float *sgto_components_z = (float*) &sgto_components_y[norbs]; //norbs
	int *sgto_powers = (int*) &sgto_components_z[norbs]; //norbs
	float *sorbital_weights = (float*) &sgto_powers[norbs]; //norbs

	float *norb_temporary = (float*) &sorbital_weights[norbs]; //[norbs x nmbody x ngauss]
	float *lmax_temporary = (float*) &norb_temporary[nmbody * ngauss * norbs]; //[(lmax+1) x nmbody x ngauss]

	float *norb_deriv_temporary = (float*) &lmax_temporary[nmbody * ngauss * (lmax + 1)]; //[norbs x nmbody x ngauss]
	float *lmax_deriv_temporary = (float*) &norb_deriv_temporary[nmbody * ngauss * norbs]; //[(lmax+1) x nmbody x ngauss]

	float *slattice_vecs = (float*) &lmax_deriv_temporary[(lmax + 1) * nmbody * ngauss];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	/*Shared Memory Definitions*/

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	float rix = coordinates[molID][iatom][0];
	float riy = coordinates[molID][iatom][1];
	float riz = coordinates[molID][iatom][2];

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x < 3) {
			for (int j = 0; j < 3; j++) {
				slattice_vecs[threadIdx.x * 3 + j] = cell[molID][threadIdx.x][j];
				sinv_lattice_vecs[threadIdx.x * 3 + j] = inv_cell[molID][threadIdx.x][j];
			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < norbs; i += blockDim.x) {
		sgto_components_x[i] = gto_components[i][0];
		sgto_components_y[i] = gto_components[i][1];
		sgto_components_z[i] = gto_components[i][2];

		sorbital_weights[i] = orbital_weights[i];

		sgto_powers[i] = gto_powers[i];
	}

	__syncthreads();

	for (int i = threadIdx.x; i < ngauss; i += blockDim.x) {
		sgridpoints[i] = gridpoints[i];
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		for (int j = 0; j < nspecies; j++) {
			for (int k = j; k < nspecies; k++) {
				smbodylist[j * nspecies + k] = mbodylist[j][k];
				smbodylist[k * nspecies + j] = smbodylist[j * nspecies + k];
			}
		}
	}

	__syncthreads();

// zero out shared data storage
	for (int i = threadIdx.x; i < krepsize; i += blockDim.x) {
		norb_temporary[i] = 0.0;
		norb_deriv_temporary[i] = 0.0;

		if (i < lrepsize) {
			lmax_temporary[i] = 0.0;
			lmax_deriv_temporary[i] = 0.0;
		}
	}

	__syncthreads();

	double sqrt_eta = sqrt(eta / M_PI);

//need to generate the representation partially first.

	float drij[3];

	for (int jatom = 0; jatom < nneighbours_i; jatom++) {

		int j = neighbourlist[molID][iatom][jatom];

		if (j == -1) {
			continue;
		}

		float rjx = coordinates[molID][j][0];
		float rjy = coordinates[molID][j][1];
		float rjz = coordinates[molID][j][2];
		int element_type = element_types[molID][j];

		drij[0] = rix - rjx;
		drij[1] = riy - rjy;
		drij[2] = riz - rjz;

		if (pbc) {
			get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
		}

		float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

		float cut = 0.5 * (cos(rij * M_PI / rcut) + 1.0);

		for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

			int z = k % ngauss;
			int korb = int(floor(float(k) / ngauss));

			int gto_power = sgto_powers[korb];

			float gto_component_x = sgto_components_x[korb];
			float gto_component_y = sgto_components_y[korb];
			float gto_component_z = sgto_components_z[korb];

			float ang = pow(drij[0], gto_component_x) * pow(drij[1], gto_component_y) * pow(drij[2], gto_component_z);

			float val = sqrt_eta * (1.0 / pow(rij, inv_factors[gto_power] + gto_power)) * ang * cut;

			float gval = exp(-eta * pow(rij - sgridpoints[z], 2.0)) * val;

			for (int m = 0; m < nspecies; m++) {

				int ej = smbodylist[element_type * nspecies + m];

				int kidx = korb * nmbody * ngauss + ej * ngauss + z;

				norb_temporary[kidx] += gval;
			}
		}
	}
	__syncthreads();

//now lets do the derivative
	for (int jatom = 0; jatom < nneighbours_i; jatom++) {

		int j = neighbourlist[molID][iatom][jatom];

		if (j == -1) {
			continue;
		}

		float rjx = coordinates[molID][j][0];
		float rjy = coordinates[molID][j][1];
		float rjz = coordinates[molID][j][2];
		int element_type = element_types[molID][j];

		drij[0] = rix - rjx;
		drij[1] = riy - rjy;
		drij[2] = riz - rjz;

		if (pbc) {
			get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
		}

		float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

		float cut = 0.5 * (cos(rij * M_PI / rcut) + 1.0);
		float dcut = -0.5 * (sin(rij * M_PI / rcut)) * M_PI / rcut;

		for (int x = 0; x < 3; x++) {

			// zero out shared data storage
			for (int i = threadIdx.x; i < krepsize; i += blockDim.x) {
				norb_deriv_temporary[i] = 0.0;
			}

			for (int i = threadIdx.x; i < lrepsize; i += blockDim.x) {
				lmax_deriv_temporary[i] = 0.0;
			}

			__syncthreads();

			float drijx = drij[x] / rij;

			for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

				int z = k % ngauss;
				int korb = int(floor(float(k) / ngauss));

				int gto_power = sgto_powers[korb];

				float gto_component_x = sgto_components_x[korb];
				float gto_component_y = sgto_components_y[korb];
				float gto_component_z = sgto_components_z[korb];

				float rscaling = (1.0 / pow(rij, inv_factors[gto_power] + gto_power));
				float drscaling = -(inv_factors[gto_power] + float(gto_power)) * (1.0 / pow(rij, 1.0 + inv_factors[gto_power] + float(gto_power)));

				float ang = pow(drij[0], gto_component_x) * pow(drij[1], gto_component_y) * pow(drij[2], gto_component_z);

				float dang[3];

				dang[0] = 0.0;
				dang[1] = 0.0;
				dang[2] = 0.0;

				//TODO need to make the following more efficient
				if (x == 0) {
					if (gto_component_x == 1.0) {
						dang[0] = -1.0 * pow(drij[1], gto_component_y) * pow(drij[2], gto_component_z);
					} else if (gto_component_x > 1.0) {
						dang[0] = -gto_component_x * pow(drij[0], gto_component_x - 1.0) * pow(drij[1], gto_component_y) * pow(drij[2], gto_component_z);
					}
				} else if (x == 1) {
					if (gto_component_y == 1.0) {
						dang[1] = -1.0 * pow(drij[0], gto_component_x) * pow(drij[2], gto_component_z);
					} else if (gto_component_y > 1.0) {
						dang[1] = -gto_component_y * pow(drij[1], gto_component_y - 1.0) * pow(drij[0], gto_component_x) * pow(drij[2], gto_component_z);
					}
				} else {
					if (gto_component_z == 1.0) {
						dang[2] = -1.0 * pow(drij[0], gto_component_x) * pow(drij[1], gto_component_y);
					} else if (gto_component_z > 1.0) {
						dang[2] = -gto_component_z * pow(drij[2], gto_component_z - 1.0) * pow(drij[0], gto_component_x) * pow(drij[1], gto_component_y);
					}
				}

				float drscalingx = drscaling * -drijx * ang * cut;
				float dangx = rscaling * dang[x] * cut;
				float dcutx = rscaling * ang * dcut * -drijx;

				float radial = exp(-eta * pow(rij - sgridpoints[z], 2.0));
				float dradial = -2.0 * eta * (rij - sgridpoints[z]);

				float val = sqrt_eta * (1.0 / pow(rij, 2.0 + gto_power)) * ang * cut;

				float gval = radial * val;

				float dval_x = sqrt_eta * radial * (drscalingx + dangx + dcutx + (rscaling * ang * cut * dradial * -drijx));

				for (int m = 0; m < nspecies; m++) {

					int ej = smbodylist[element_type * nspecies + m];

					int kidx = korb * nmbody * ngauss + ej * ngauss + z;

					norb_deriv_temporary[kidx] += dval_x;
				}
			}

			__syncthreads();

			//contract derivative into (lmax+1) channels
			for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

				int z = k % ngauss;
				int korb = int(floor(float(k) / ngauss));

				int lchannel = sgto_powers[korb];

				for (int m = 0; m < nspecies; m++) {
					for (int n = m; n < nspecies; n++) {

						int mnidx = smbodylist[m * nspecies + n];

						int kmn = korb * nmbody * ngauss + mnidx * ngauss + z;
						int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

						float val = norb_temporary[kmn];
						float val_deriv = norb_deriv_temporary[kmn];

						atomicAdd(&lmax_deriv_temporary[lmn], lchannel_weights[lchannel] * sorbital_weights[korb] * 2.0 * val * val_deriv);
					}
				}
			}

			__syncthreads();

			//subtract single-element contributions
			for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

				int z = k % ngauss;
				int l = int(floor(float(k) / ngauss));

				for (int m = 0; m < nspecies; m++) {
					for (int n = m + 1; n < nspecies; n++) {

						int mnidx = smbodylist[m * nspecies + n];
						int mmidx = smbodylist[m * nspecies + m];
						int nnidx = smbodylist[n * nspecies + n];

						int lmn = l * nmbody * ngauss + mnidx * ngauss + z;
						int lmm = l * nmbody * ngauss + mmidx * ngauss + z;
						int lnn = l * nmbody * ngauss + nnidx * ngauss + z;

						float t1 = lmax_deriv_temporary[lmm];
						float t2 = lmax_deriv_temporary[lnn];

						lmax_deriv_temporary[lmn] -= (t1 + t2);
					}
				}
			}

			__syncthreads();

			//save to global memory - will be somewhat sparse, TODO - can probably reduce the cost of this
			// nspecies * (lmax+1) * ngaussians should be nonzero vs (nspecies +1)/2 * nspecies * (lmax+1) * ngaussians

			for (int k = threadIdx.x; k < lrepsize; k += blockDim.x) {
				grad[molID][iatom][j][x][k] += lmax_deriv_temporary[k];
				grad[molID][iatom][iatom][x][k] -= lmax_deriv_temporary[k];
			}
		}
	}

//now back to atomic representation...

//contract representation into (lmax+1) channels

//norb_temporary:[norbs, nmbody, ngauss]

	for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int korb = int(floor(float(k) / ngauss));

		int lchannel = sgto_powers[korb];

		for (int m = 0; m < nspecies; m++) {
			for (int n = m; n < nspecies; n++) {

				int mnidx = smbodylist[m * nspecies + n];

				int kmn = korb * nmbody * ngauss + mnidx * ngauss + z;
				int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

				float val = norb_temporary[kmn];

				atomicAdd(&lmax_temporary[lmn], lchannel_weights[lchannel] * sorbital_weights[korb] * val * val);
			}
		}
	}
	__syncthreads();

//subtract single-element contributions
	for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int l = int(floor(float(k) / ngauss));

		for (int m = 0; m < nspecies; m++) {
			for (int n = m + 1; n < nspecies; n++) {

				int mnidx = smbodylist[m * nspecies + n];
				int mmidx = smbodylist[m * nspecies + m];
				int nnidx = smbodylist[n * nspecies + n];

				int lmn = l * nmbody * ngauss + mnidx * ngauss + z;
				int lmm = l * nmbody * ngauss + mmidx * ngauss + z;
				int lnn = l * nmbody * ngauss + nnidx * ngauss + z;

				float t1 = lmax_temporary[lmm];
				float t2 = lmax_temporary[lnn];

				lmax_temporary[lmn] -= (t1 + t2);
			}
		}
	}

	__syncthreads();

//save to global memory
	for (int k = threadIdx.x; k < lrepsize; k += blockDim.x) {
		output[molID][iatom][k] = lmax_temporary[k];
	}
}

__global__
void get_element_types_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> natom_counts,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types) {

	int natoms = natom_counts[blockIdx.x];
	int nspecies = species.size(0);

	for (int iatom = threadIdx.x; iatom < natoms; iatom += blockDim.x) {

		if (iatom < natoms) {

			int qi = charges[blockIdx.x][iatom];

			int index = -1;
			for (int j = 0; j < nspecies; j++) {
				if (qi == species[j]) {
					index = j;
				}
			}

			element_types[blockIdx.x][iatom] = index;
		}
	}
}

void getElementTypesCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor natom_counts, torch::Tensor species, torch::Tensor element_types) {

	int nbatch = coordinates.size(0);
	const int nthreads = 32;

	get_element_types_kernel<<<nbatch, nthreads>>>(coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void EGTOCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor mbodylist, torch::Tensor gto_components,
		torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, torch::Tensor lchannel_weights, torch::Tensor inv_factors, float eta,
		int lmax, float rcut, torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor gto_output) {

	const int nthreads = 32;

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int(float((nspecies + 1) / 2.0) * nspecies);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	//printf("nblocks: %d, Max neighbours: %d norbs: %d, nmbody: %d ngaussians: %d\n", currBatch, max_neighbours, norbs, nmbody, ngaussians);

	int shared_mem_size = 2 * nspecies + ngaussians + 5 * norbs + (norbs * nmbody * ngaussians) + (lmax + 1) * nmbody * ngaussians + 18;

	//printf("Shared mem requested: %d bytes\n", shared_mem_size);
	egto_atomic_representation_cuda<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			mbodylist.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			gto_components.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			orbital_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			gto_powers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			gridpoints.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			lchannel_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			inv_factors.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), eta, lmax,
			rcut,
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void EGTODerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor mbodylist, torch::Tensor gto_components,
		torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, torch::Tensor lchannel_weights, torch::Tensor inv_factors, float eta,
		int lmax, float rcut, torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor gto_output, torch::Tensor gto_output_derivative) {

	const int nthreads = 32;

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int(float((nspecies + 1) / 2.0) * nspecies);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	int shared_mem_size = 2 * nspecies + ngaussians + 5 * norbs + 2 * ((norbs * nmbody * ngaussians) + (lmax + 1) * nmbody * ngaussians) + 18;

	egto_atomic_representation_derivative_cuda<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			mbodylist.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			gto_components.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			orbital_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			gto_powers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			gridpoints.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			lchannel_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			inv_factors.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), eta, lmax,
			rcut, cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), gto_output_derivative.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

