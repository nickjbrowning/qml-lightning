#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

#define GAUSSIAN_DISTRIBUTION 0
#define LOGNORMAL_DISTRIBUTION 1
#define EXPEXP_DISTRIBUTION 2

#define COSINE_CUTOFF 0
#define SWITCH_FUNCTION 1

#define SQRT2PI 2.506628275f

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

__device__ float get_cutoff(float rij, float rcut, float rswitch, int cutoff_type) {
	float cut = 1.0;

	switch (cutoff_type) {

	case COSINE_CUTOFF:
		cut = 0.5 * (cos(rij * M_PI / rcut) + 1.0);
		break;

	case SWITCH_FUNCTION:
		if (rij > rswitch) {
			float sx = (rij - rswitch) / (rcut - rswitch);
			cut = cut - 6.0 * powf(sx, 5.0) + 15.0 * powf(sx, 4.0) - 10.0 * powf(sx, 3.0);
		}
		break;
	default:
		cut = 0.5 * (cos(rij * M_PI / rcut) + 1.0);
		break;
	}

	return cut;
}

__device__ float get_cutoff_derivative(float rij, float rcut, float rswitch, int cutoff_type) {
	float dcut = 0.0;

	switch (cutoff_type) {

	case COSINE_CUTOFF:

		dcut = -0.5 * (sin(rij * M_PI / rcut)) * M_PI / rcut;
		break;

	case SWITCH_FUNCTION:

		if (rij > rswitch) {
			float sx = (rij - rswitch) / (rcut - rswitch);
			dcut = (1.0 / (rcut - rswitch)) * (-30.0 * powf(sx, 4.0) + 60.0 * powf(sx, 3.0) - 30.0 * powf(sx, 2.0));
		}
		break;

	default:
		dcut = -0.5 * (sin(rij * M_PI / rcut)) * M_PI / rcut;
		break;
	}

	return dcut;
}

__device__ float get_radial_distribution(float rij, float eta, float *gridpoints, int index, int distribution_type) {

	float d = 0.0;
	float mu = 0.0;
	float sigma2 = 0.0;
	float sigma = 0.0;

	switch (distribution_type) {

	case GAUSSIAN_DISTRIBUTION:
		d = sqrt(eta / M_PI) * exp(-eta * powf(rij - gridpoints[index], 2.0));
		//d = exp(-eta * powf(rij - gridpoints[index], 2.0));
		break;

	case LOGNORMAL_DISTRIBUTION:
		mu = log(rij / sqrt(1.0 + (eta / powf(rij, 2.0))));
		sigma2 = log(1.0 + (eta / powf(rij, 2.0)));
		sigma = sqrt(sigma2);

		d = (1.0 / (gridpoints[index] * sigma * SQRT2PI)) * exp(-powf(log(gridpoints[index]) - mu, 2.0) / (2.0 * sigma2));
		break;

	case EXPEXP_DISTRIBUTION:
		d = exp(-eta * powf(exp(-rij) - gridpoints[index], 2.0));
		break;

	default:
		d = sqrt(eta / M_PI) * exp(-eta * powf(rij - gridpoints[index], 2.0));
		//d = exp(-eta * powf(rij - gridpoints[index], 2.0));
		break;
	}

	return d;
}

__device__ float get_radial_derivative_distribution(float drijx, float rij, float eta, float *gridpoints, int index, int distribution_type) {

	float dradial_dx = 0.0;

	float sqrt_eta = sqrt(eta / M_PI);
	float mu = 0.0;
	float sigma = 0.0;
	float sigma2 = 0.0;
	float sigma4 = 0.0;
	float lnRs = 0.0;
	float exp_ln = 0.0;
	float rij2 = 0.0;
	float dmu_dx = 0.0;
	float dsigma_dx = 0.0;

	switch (distribution_type) {

	case GAUSSIAN_DISTRIBUTION:
		dradial_dx = sqrt_eta * exp(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		//dradial_dx = exp(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		break;

	case LOGNORMAL_DISTRIBUTION:

		mu = log(rij / sqrt(1.0 + (eta / powf(rij, 2.0))));
		sigma = sqrt(log(1.0 + (eta / powf(rij, 2.0))));
		sigma2 = powf(sigma, 2.0);
		sigma4 = powf(sigma, 4.0);
		lnRs = log(gridpoints[index]);
		exp_ln = exp(-powf(lnRs - mu, 2.0) / powf(sigma, 2.0) * 0.5);
		rij2 = powf(rij, 2.0);

		dsigma_dx = drijx * eta * (1.0 / ((eta + rij2) * rij * sqrt(log((eta + rij2) / rij2))));
		dmu_dx = -drijx * ((2 * eta + rij2) / ((eta + rij2) * rij));

		dradial_dx = (sqrt(2.0) / (2 * sqrt(M_PI) * gridpoints[index] * sigma4))
				* (((mu - lnRs) * dsigma_dx - sigma * dmu_dx) * (mu - lnRs) - sigma2 * dsigma_dx) * exp_ln;

		break;

	case EXPEXP_DISTRIBUTION:
		dradial_dx = 2.0 * eta * (-gridpoints[index] + exp(-rij)) * exp(-eta * powf(exp(-rij) - gridpoints[index], 2.0)) * exp(-rij) * -drijx;
		break;

	default:
		dradial_dx = sqrt_eta * exp(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		//dradial_dx = exp(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		break;
	}

	return dradial_dx;
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
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> inv_factors, float eta, int lmax, float rcut, float rswitch,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell, int cutoff_type, int distribution_type,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int ngauss = gridpoints.size(0);

	int norbs = gto_components.size(0);
	int nspecies = species.size(0);

	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *sgridpoints = (float*) &s; //ngaussians
	int *smbodylist = (int*) &sgridpoints[ngauss]; //nspecies * nspecies
	float *sgto_components_x = (float*) &smbodylist[nspecies * nspecies]; //norbs
	float *sgto_components_y = (float*) &sgto_components_x[norbs]; //norbs
	float *sgto_components_z = (float*) &sgto_components_y[norbs]; //norbs
	int *sgto_powers = (int*) &sgto_components_z[norbs]; //norbs
	float *sorbital_weights = (float*) &sgto_powers[norbs]; //norbs

	float *lmax_temporary = (float*) &sorbital_weights[norbs]; //[(lmax+1) x nmbody x ngauss]

	float *slattice_vecs = (float*) &lmax_temporary[(lmax + 1) * nmbody * ngauss];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	float *local_rep = (float*) &sinv_lattice_vecs[9];

	float *scoords_x = (float*) &local_rep[nmbody * blockDim.x];
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];

	/*Shared Memory Definitions*/

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	float rix = coordinates[molID][iatom][0];
	float riy = coordinates[molID][iatom][1];
	float riz = coordinates[molID][iatom][2];

	float sqrt_eta = sqrt(eta / M_PI);

	bool pbc = false;

	/*Each thread only stores the m-body components from the uncontracted GTO representation locally. The full
	 * uncontacted GTO representation is not built (unlike in egto_atomic_representation_cuda).
	 * Results in significantly reduced shared memory footprint, as only the final contracted representation is stored.*/

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x < 3) {
			for (int j = 0; j < 3; j++) {
				slattice_vecs[threadIdx.x * 3 + j] = cell[molID][threadIdx.x][j];
				sinv_lattice_vecs[threadIdx.x * 3 + j] = inv_cell[molID][threadIdx.x][j];
			}
		}
	}

	for (int i = threadIdx.x; i < norbs; i += blockDim.x) {
		sgto_components_x[i] = gto_components[i][0];
		sgto_components_y[i] = gto_components[i][1];
		sgto_components_z[i] = gto_components[i][2];

		sorbital_weights[i] = orbital_weights[i];

		sgto_powers[i] = gto_powers[i];
	}

	for (int i = threadIdx.x; i < ngauss; i += blockDim.x) {
		sgridpoints[i] = gridpoints[i];
	}

	if (threadIdx.x == 0) {
		for (int j = 0; j < nspecies; j++) {
			for (int k = j; k < nspecies; k++) {
				smbodylist[j * nspecies + k] = mbodylist[j][k];
				smbodylist[k * nspecies + j] = mbodylist[k][j];

			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < lrepsize; i += blockDim.x) {
		lmax_temporary[i] = 0.0;
	}

//load coordinates into shared memory
	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];

	}

	__syncthreads();

	float drij[3];

	for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

		for (int i = threadIdx.x; i < nmbody * blockDim.x; i += blockDim.x) {
			local_rep[i] = 0.0;
		}

		int z = k % ngauss;
		int korb = k / ngauss;

		int gto_power = sgto_powers[korb];
		int lchannel = sgto_powers[korb];

		float inv_factor = inv_factors[gto_power];

		float gto_component_x = sgto_components_x[korb];
		float gto_component_y = sgto_components_y[korb];
		float gto_component_z = sgto_components_z[korb];

		float lchannel_weight = lchannel_weights[lchannel];
		float orbital_weight = sorbital_weights[korb];

		for (int jatom = 0; jatom < nneighbours_i; jatom++) {

			float rjx = scoords_x[jatom];
			float rjy = scoords_y[jatom];
			float rjz = scoords_z[jatom];
			int element_type = selement_types[jatom];

			drij[0] = rix - rjx;
			drij[1] = riy - rjy;
			drij[2] = riz - rjz;

			if (pbc) {
				get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
			}

			float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

			float cut = get_cutoff(rij, rcut, rswitch, cutoff_type);

			float ang = powf(drij[0], gto_component_x) * powf(drij[1], gto_component_y) * powf(drij[2], gto_component_z);

			float val = (1.0 / powf(rij, inv_factor + gto_power)) * ang * cut;

			float g = get_radial_distribution(rij, eta, sgridpoints, z, distribution_type);

			float gval = g * val;

			for (int m = 0; m < nspecies; m++) {

				int mnidx = smbodylist[element_type * nspecies + m];

				local_rep[mnidx * blockDim.x + threadIdx.x] += gval;

			}
		}

		//contract into lmax channels here
		for (int m = 0; m < nspecies; m++) {
			for (int n = m; n < nspecies; n++) {

				int mnidx = smbodylist[m * nspecies + n];

				int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

				float val = local_rep[mnidx * blockDim.x + threadIdx.x];

				atomicAdd(&lmax_temporary[lmn], lchannel_weight * orbital_weight * val * val);
			}
		}
	}

	__syncthreads();

//subtract single-element contributions
	for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int l = k / ngauss;

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

__global__
void egto_atomic_representation_derivative_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
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
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> inv_factors, float eta, int lmax, float rcut, float rswitch,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell, int cutoff_type, int distribution_type,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output, torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> grad) {

	extern __shared__ int s[];

	int nbatch = coordinates.size(0);
	int ngauss = gridpoints.size(0);
	int norbs = gto_components.size(0);
	int nspecies = species.size(0);

	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *sgridpoints = (float*) &s; //ngaussians
	int *smbodylist = (int*) &sgridpoints[ngauss]; //nspecies * nspecies
	float *sgto_components_x = (float*) &smbodylist[nspecies * nspecies]; //norbs
	float *sgto_components_y = (float*) &sgto_components_x[norbs]; //norbs
	float *sgto_components_z = (float*) &sgto_components_y[norbs]; //norbs
	int *sgto_powers = (int*) &sgto_components_z[norbs]; //norbs
	float *sorbital_weights = (float*) &sgto_powers[norbs]; //norbs

	float *lmax_temporary = (float*) &sorbital_weights[norbs];

	float *slattice_vecs = (float*) &lmax_temporary[(lmax + 1) * nmbody * ngauss];
	float *sinv_lattice_vecs = (float*) &slattice_vecs[9];

	float *local_rep = (float*) &sinv_lattice_vecs[9];

	float *scoords_x = (float*) &local_rep[nmbody * blockDim.x];
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];
	int *sneighbours = (int*) &selement_types[max_neighbours];

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

//	if (threadIdx.x == 0) {
//		for (int j = 0; j < nspecies; j++) {
//			for (int k = j; k < nspecies; k++) {
//				smbodylist[j * nspecies + k] = mbodylist[j][k];
//				smbodylist[k * nspecies + j] = mbodylist[k][j];
//			}
//		}
//	}

	__syncthreads();

// zero out shared data storage
	for (int i = threadIdx.x; i < lrepsize; i += blockDim.x) {
		lmax_temporary[i] = 0.0;
	}

	__syncthreads();

//load coordinates into shared memory
	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];
		sneighbours[jatom] = j;

	}

	__syncthreads();

	float sqrt_eta = sqrt(eta / M_PI);

//need to generate the representation partially first.

	float drij[3];
	float drijx[3];
	float dang[3];

	for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

		for (int i = 0; i < nspecies; i++) {
			for (int j = i; j < nspecies; j++) {
				int mnidx = mbodylist[i][j];
				local_rep[mnidx * blockDim.x + threadIdx.x] = 0.0;
			}
		}

		int z = k % ngauss;
		int korb = k / ngauss;

		int gto_power = sgto_powers[korb];
		int lchannel = sgto_powers[korb];

		float inv_factor = inv_factors[lchannel];

		float gto_component_x = sgto_components_x[korb];
		float gto_component_deriv_x = (gto_component_x == 1.0) ? -1.0 : -gto_component_x;

		float gto_component_y = sgto_components_y[korb];
		float gto_component_deriv_y = (gto_component_y == 1.0) ? -1.0 : -gto_component_y;

		float gto_component_z = sgto_components_z[korb];
		float gto_component_deriv_z = (gto_component_z == 1.0) ? -1.0 : -gto_component_z;

		float lchannel_weight = lchannel_weights[lchannel];
		float orbital_weight = sorbital_weights[korb];

		for (int jatom = 0; jatom < nneighbours_i; jatom++) {

			float rjx = scoords_x[jatom];
			float rjy = scoords_y[jatom];
			float rjz = scoords_z[jatom];
			int element_type = selement_types[jatom];

			drij[0] = rix - rjx;
			drij[1] = riy - rjy;
			drij[2] = riz - rjz;

			if (pbc) {
				get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
			}

			float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

			float cut = get_cutoff(rij, rcut, rswitch, cutoff_type);

			float ang = powf(drij[0], gto_component_x) * powf(drij[1], gto_component_y) * powf(drij[2], gto_component_z);

			float val = (1.0 / powf(rij, inv_factor + gto_power)) * ang * cut;

			float g = get_radial_distribution(rij, eta, sgridpoints, z, distribution_type);

			float gval = g * val;

			for (int m = 0; m < nspecies; m++) {

				int mnidx = mbodylist[element_type][m];

				local_rep[mnidx * blockDim.x + threadIdx.x] += gval;

			}
		}

		//contract representation into lmax channels here
		for (int m = 0; m < nspecies; m++) {
			for (int n = m; n < nspecies; n++) {

				int mnidx = mbodylist[m][n];

				int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

				float val = local_rep[mnidx * blockDim.x + threadIdx.x];

				atomicAdd(&lmax_temporary[lmn], lchannel_weight * orbital_weight * val * val);
			}
		}

		//derivatives
		for (int jatom = 0; jatom < nneighbours_i; jatom++) {

			float rjx = scoords_x[jatom];
			float rjy = scoords_y[jatom];
			float rjz = scoords_z[jatom];
			int element_type = selement_types[jatom];
			int j = sneighbours[jatom];

			drij[0] = rix - rjx;
			drij[1] = riy - rjy;
			drij[2] = riz - rjz;

			if (pbc) {
				get_pbc_dij(drij, slattice_vecs, sinv_lattice_vecs);
			}

			float rij = sqrt(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

			float cut = get_cutoff(rij, rcut, rswitch, cutoff_type);
			float dcut = get_cutoff_derivative(rij, rcut, rswitch, cutoff_type);

			float rscaling = (1.0 / powf(rij, inv_factor + gto_power));
			float drscaling = -(inv_factor + float(gto_power)) * (1.0 / powf(rij, 1.0 + inv_factor + float(gto_power)));
			float ang = powf(drij[0], gto_component_x) * powf(drij[1], gto_component_y) * powf(drij[2], gto_component_z);

			dang[0] = 0.0;
			dang[1] = 0.0;
			dang[2] = 0.0;

			if (gto_component_x >= 1)
				dang[0] = gto_component_deriv_x * powf(drij[0], (int) gto_component_x - 1) * powf(drij[1], (int) gto_component_y)
						* powf(drij[2], (int) gto_component_z);
			if (gto_component_y >= 1)
				dang[1] = gto_component_deriv_y * powf(drij[1], (int) gto_component_y - 1) * powf(drij[0], (int) gto_component_x)
						* powf(drij[2], (int) gto_component_z);
			if (gto_component_z >= 1)
				dang[2] = gto_component_deriv_z * powf(drij[2], (int) gto_component_z - 1) * powf(drij[0], (int) gto_component_x)
						* powf(drij[1], (int) gto_component_y);

			drijx[0] = drij[0] / rij;
			drijx[1] = drij[1] / rij;
			drijx[2] = drij[2] / rij;

			float radial = get_radial_distribution(rij, eta, sgridpoints, z, distribution_type);

			//float radial = sqrt_eta * exp(-eta * powf(rij - sgridpoints[z], 2.0));
			//float dradial = -2.0 * eta * (rij - sgridpoints[z]) * radial;

			/*if (iatom == 2 && j == 0 && z == 0) {
			 printf("---jatom: %i, neighbour: %i, drij: %.5e %.5e %.5e, blockDim: %i, thread: %i---\n", jatom, j, drij[0], drij[1], drij[2], blockDim.x,
			 threadIdx.x);
			 printf("ang: %.5e, pows: %.5e %.5e %.5e, components: %.2e %.2e %.2e\n", ang, powf(drij[0], gto_component_x), powf(drij[1], gto_component_y),
			 powf(drij[2], gto_component_z), gto_component_x, gto_component_y, gto_component_z);
			 }*/
			for (int x = 0; x < 3; x++) {

				/*if (iatom == 2 && j == 0 && x == 0 && z == 0) {
				 printf("coordinate: %i\n", x);
				 }*/

				float dradial = get_radial_derivative_distribution(drijx[x], rij, eta, sgridpoints, z, distribution_type);

				float drscalingx = drscaling * -drijx[x] * ang * radial * cut;
				float dangx = rscaling * dang[x] * radial * cut;
				float dcutx = rscaling * ang * radial * dcut * -drijx[x];
				//float dradialx = rscaling * ang * dradial * -drijx[x] * cut;

				float dradialx = rscaling * ang * dradial * cut;

				//float dval_x = radial * (drscalingx + dangx + dcutx + (rscaling * ang * cut * dradial * -drijx[x]));

				float deriv = lchannel_weight * orbital_weight * 2.0 * (drscalingx + dangx + dradialx + dcutx);

				/*if (iatom == 2 && j == 0 && x == 0 && z == 0) {

				 printf(
				 "lchannel %i, lchannel_weight %.5e, orb %i, gauss: %i weight: %f, gto_component_x: %f, gto_component_deriv_x: %f, drscalingx: %.5e, dangx: %.5e, dradialx: %.5e, dcutx: %.5e\n",
				 lchannel, lchannel_weight, korb, z, orbital_weight, gto_component_x, gto_component_deriv_x, drscalingx, ang, dangx, dradialx,
				 dcutx);
				 }*/
				for (int n = 0; n < nspecies; n++) {

					int mnidx = mbodylist[element_type][n];

					int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

					float val = local_rep[mnidx * blockDim.x + threadIdx.x];

					/*if (iatom == 2 && j == 0 && x == 0 && z == 0) {
					 printf("%.5e, %.5e, %.5e\n", val, deriv, deriv * val);
					 }*/

					float final = deriv * val;

					//atomicAdd(&lmax_deriv_temporary[lmn], lchannel_weight * orbital_weight * 2.0 * val * dval_x);

					atomicAdd(&grad[molID][iatom][j][x][lmn], final);
					atomicAdd(&grad[molID][iatom][iatom][x][lmn], -final);
				}
			}
		}
	}

	__syncthreads();

//subtract single-element contributions
	for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

		int z = k % ngauss;
		int l = k / ngauss;

		for (int m = 0; m < nspecies; m++) {
			for (int n = m + 1; n < nspecies; n++) {

				int mnidx = mbodylist[m][n];
				int mmidx = mbodylist[m][m];
				int nnidx = mbodylist[n][n];

				int lmn = l * nmbody * ngauss + mnidx * ngauss + z;
				int lmm = l * nmbody * ngauss + mmidx * ngauss + z;
				int lnn = l * nmbody * ngauss + nnidx * ngauss + z;

				float t1 = lmax_temporary[lmm];
				float t2 = lmax_temporary[lnn];

				atomicAdd(&lmax_temporary[lmn], -(t1 + t2));

				for (int x = 0; x < 3; x++) {
					float t3 = grad[molID][iatom][iatom][x][lmm];
					float t4 = grad[molID][iatom][iatom][x][lnn];

					atomicAdd(&grad[molID][iatom][iatom][x][lmn], -(t3 + t4));
				}
			}
		}
	}

	__syncthreads();

	for (int jatom = 0; jatom < nneighbours_i; jatom++) {

		int j = sneighbours[jatom];

		for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

			int z = k % ngauss;
			int l = k / ngauss;

			for (int m = 0; m < nspecies; m++) {
				for (int n = m + 1; n < nspecies; n++) {

					int mnidx = mbodylist[m][n];
					int mmidx = mbodylist[m][m];
					int nnidx = mbodylist[n][n];

					int lmn = l * nmbody * ngauss + mnidx * ngauss + z;
					int lmm = l * nmbody * ngauss + mmidx * ngauss + z;
					int lnn = l * nmbody * ngauss + nnidx * ngauss + z;

					for (int x = 0; x < 3; x++) {

						float t3 = grad[molID][iatom][j][x][lmm];
						float t4 = grad[molID][iatom][j][x][lnn];

						atomicAdd(&grad[molID][iatom][j][x][lmn], -(t3 + t4));
					}
				}
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
		int lmax, float rcut, float rswitch, torch::Tensor cell, torch::Tensor inv_cell, int cutoff_type, int distribution_type, torch::Tensor gto_output) {

	const int nthreads = 32;

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

//printf("nblocks: %d, Max neighbours: %d norbs: %d, nmbody: %d ngaussians: %d\n", currBatch, max_neighbours, norbs, nmbody, ngaussians);

	int shared_mem_size = nspecies * nspecies + ngaussians + 5 * norbs + nmbody * nthreads + (lmax + 1) * nmbody * ngaussians + 18 + 4 * max_neighbours;

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
			rcut, rswitch,
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), cutoff_type, distribution_type,
			gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void EGTODerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor mbodylist, torch::Tensor gto_components,
		torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, torch::Tensor lchannel_weights, torch::Tensor inv_factors, float eta,
		int lmax, float rcut, float rswitch, torch::Tensor cell, torch::Tensor inv_cell, int cutoff_type, int distribution_type, torch::Tensor gto_output,
		torch::Tensor gto_output_derivative) {

	const int nthreads = 32;

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int((float(nspecies + 1.0) / 2.0) * nspecies);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

//int shared_mem_size = nspecies * nspecies + ngaussians + 5 * norbs + nmbody * nthreads + (lmax + 1) * nmbody * ngaussians + 18 + 4 * max_neighbours;

	int shared_mem_size = nspecies * nspecies + ngaussians + 5 * norbs + nmbody * nthreads + (lmax + 1) * nmbody * ngaussians + 18 + 5 * max_neighbours;

//printf("CURR BATCH: %i, NTHREADS: %i\n", currBatch, nthreads);

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
			rcut, rswitch, cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), cutoff_type, distribution_type,
			gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), gto_output_derivative.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

