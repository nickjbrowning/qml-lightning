#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

#define FULL_MASK 0xffffffff

/*Reasonably efficient implementation - builds the uncontracted GTO basis in shared memory, and then contracts it in a separate
 *shared memory buffer, before saving to global memory. Parallelisation occurs over norbs x gridpoints, rather than natoms, as
 *this has better potential for removing bank conflicts and serialising global memory access. */

__global__ void EGTO_atomic_kernel_gridpara_float(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mbodylist,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gto_components,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> orbital_weights,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> gto_powers,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gridpoints, float eta, int lmax, float rcut,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int nbatch = coordinates.size(0);
	int ngauss = gridpoints.size(0);
	int natoms = coordinates.size(1);
	int norbs = gto_components.size(0);
	int nspecies = species.size(0);
	int nmbody = int(((float(nspecies) + 1.0) / 2.0) * nspecies);

	int krepsize = nmbody * norbs * ngauss;
	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *scoordinates_x = (float*) &s; //natoms
	float *scoordinates_y = (float*) &scoordinates_x[natoms]; //natoms
	float *scoordinates_z = (float*) &scoordinates_y[natoms]; //natoms
	float *scharges = (float*) &scoordinates_z[natoms]; //natoms
	int *selement_types = (int*) &scharges[natoms]; //natoms
	float *sgridpoints = (float*) &selement_types[natoms]; //ngaussians
	int *smbodylist = (int*) &sgridpoints[ngauss]; //nspecies * nspecies
	float *sgto_components_x = (float*) &smbodylist[nspecies * nspecies]; //norbs
	float *sgto_components_y = (float*) &sgto_components_x[norbs]; //norbs
	float *sgto_components_z = (float*) &sgto_components_y[norbs]; //norbs
	int *sgto_powers = (int*) &sgto_components_z[norbs]; //norbs
	float *sorbital_weights = (float*) &sgto_powers[norbs]; //norbs

	float *norb_temporary = (float*) &sorbital_weights[norbs]; //[norbs x nmbody x ngauss]
	float *lmax_temporary = (float*) &norb_temporary[nmbody * ngauss * norbs]; //[(lmax+1) x nmbody x ngauss]
	/*Shared Memory Definitions*/

	int batchID = int(floor(float(blockIdx.x / natoms)));
	int iatom = blockIdx.x % natoms;

	for (int jatom = threadIdx.x; jatom < natoms; jatom += blockDim.x) {
		scoordinates_x[jatom] = coordinates[batchID][jatom][0];
		scoordinates_y[jatom] = coordinates[batchID][jatom][1];
		scoordinates_z[jatom] = coordinates[batchID][jatom][2];
		scharges[jatom] = charges[batchID][jatom];
		selement_types[jatom] = element_types[batchID][jatom];
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

// zero out shared data storage
	for (int i = threadIdx.x; i < krepsize; i += blockDim.x) {
		norb_temporary[i] = 0.0;

		if (i < lrepsize) {
			lmax_temporary[i] = 0.0;
		}
	}

	__syncthreads();

	float rix = scoordinates_x[iatom];
	float riy = scoordinates_y[iatom];
	float riz = scoordinates_z[iatom];

	__syncthreads();

//save norb,ngaussian points into shared memory
	for (int jatom = 0; jatom < natoms; jatom++) {

		if (iatom == jatom)
			continue;

		int element_type = selement_types[jatom];

		float rjx = scoordinates_x[jatom];
		float rjy = scoordinates_y[jatom];
		float rjz = scoordinates_z[jatom];

		float rijx = rix - rjx;
		float rijy = riy - rjy;
		float rijz = riz - rjz;

		float rij = sqrtf(rijx * rijx + rijy * rijy + rijz * rijz);

		if (rij > rcut) {
			continue;
		}

		float cut = 0.5 * (cosf(rij * M_PI / rcut) + 1.0);

		for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

			int z = k % ngauss;
			int korb = int(floor(float(k) / ngauss));

			int gto_power = sgto_powers[korb];

			float gto_component_x = sgto_components_x[korb];
			float gto_component_y = sgto_components_y[korb];
			float gto_component_z = sgto_components_z[korb];

			float ang = powf(rijx, gto_component_x) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);

			float val = sqrtf(eta / M_PI) * (1.0 / powf(rij, 2.0 + gto_power)) * ang * cut;

			float gval = expf(-eta * powf(rij - sgridpoints[z], 2.0)) * val;

			for (int m = 0; m < nspecies; m++) {

				int ej = smbodylist[element_type * nspecies + m];

				int kidx = korb * nmbody * ngauss + ej * ngauss + z;

				norb_temporary[kidx] += gval;
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
				atomicAdd(&lmax_temporary[lmn], sorbital_weights[korb] * val * val);
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
		output[batchID][iatom][k] = lmax_temporary[k];
	}
}

__global__ void EGTO_atomic_derivative_kernel_float(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mbodylist,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gto_components,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> orbital_weights,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> gto_powers,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gridpoints, float eta, int lmax, float rcut,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output, torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> grad) {

	extern __shared__ int s[];

	int nbatch = coordinates.size(0);
	int ngauss = gridpoints.size(0);
	int natoms = coordinates.size(1);
	int norbs = gto_components.size(0);
	int nspecies = species.size(0);
	int nmbody = int(((float(nspecies) + 1.0) / 2.0) * nspecies);

	int krepsize = nmbody * norbs * ngauss;
	int lrepsize = nmbody * (lmax + 1) * ngauss;

	/*Shared Memory Definitions*/
	float *scoordinates_x = (float*) &s; //natoms
	float *scoordinates_y = (float*) &scoordinates_x[natoms]; //natoms
	float *scoordinates_z = (float*) &scoordinates_y[natoms]; //natoms
	float *scharges = (float*) &scoordinates_z[natoms]; //natoms
	int *selement_types = (int*) &scharges[natoms]; //natoms
	float *sgridpoints = (float*) &selement_types[natoms]; //ngaussians
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

	/*Shared Memory Definitions*/

	int batchID = int(floor(float(blockIdx.x / natoms)));
	int iatom = blockIdx.x % natoms;

	for (int jatom = threadIdx.x; jatom < natoms; jatom += blockDim.x) {
		scoordinates_x[jatom] = coordinates[batchID][jatom][0];
		scoordinates_y[jatom] = coordinates[batchID][jatom][1];
		scoordinates_z[jatom] = coordinates[batchID][jatom][2];
		scharges[jatom] = charges[batchID][jatom];
		selement_types[jatom] = element_types[batchID][jatom];
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

	float rix = scoordinates_x[iatom];
	float riy = scoordinates_y[iatom];
	float riz = scoordinates_z[iatom];

	double sqrtf_eta = sqrt(eta / M_PI);

	__syncthreads();

//need to generate the representation partially, first, for the derivative.
	for (int jatom = 0; jatom < natoms; jatom++) {

		if (iatom == jatom)
			continue;

		int element_type = selement_types[jatom];

		float rjx = scoordinates_x[jatom];
		float rjy = scoordinates_y[jatom];
		float rjz = scoordinates_z[jatom];

		float rijx = rix - rjx;
		float rijy = riy - rjy;
		float rijz = riz - rjz;

		float rij = sqrtf(rijx * rijx + rijy * rijy + rijz * rijz);

		if (rij > rcut) {
			continue;
		}

		float cut = 0.5 * (cosf(rij * M_PI / rcut) + 1.0);

		for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

			int z = k % ngauss;
			int korb = int(floor(float(k) / ngauss));

			int gto_power = sgto_powers[korb];

			float gto_component_x = sgto_components_x[korb];
			float gto_component_y = sgto_components_y[korb];
			float gto_component_z = sgto_components_z[korb];

			float ang = powf(rijx, gto_component_x) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);

			float val = sqrtf_eta * (1.0 / powf(rij, 2.0 + gto_power)) * ang * cut;

			//float val = sqrtf_eta * (1.0 / rij) * ang * cut;

			float gval = expf(-eta * powf(rij - sgridpoints[z], 2.0)) * val;

			for (int m = 0; m < nspecies; m++) {

				int ej = smbodylist[element_type * nspecies + m];

				int kidx = korb * nmbody * ngauss + ej * ngauss + z;

				norb_temporary[kidx] += gval;
			}
		}
	}
	__syncthreads();

//now lets do the derivative
	for (int jatom = 0; jatom < natoms; jatom++) {

		if (iatom == jatom)
			continue;

		int element_type = selement_types[jatom];

		float rjx = scoordinates_x[jatom];
		float rjy = scoordinates_y[jatom];
		float rjz = scoordinates_z[jatom];

		float rijx = rix - rjx;
		float rijy = riy - rjy;
		float rijz = riz - rjz;

		float drij[3];

		drij[0] = rijx;
		drij[1] = rijy;
		drij[2] = rijz;

		float rij = sqrtf(rijx * rijx + rijy * rijy + rijz * rijz);

		if (rij > rcut) {
			continue;
		}

		float cut = 0.5 * (cosf(rij * M_PI / rcut) + 1.0);
		float dcut = -0.5 * (sinf(rij * M_PI / rcut)) * M_PI / rcut;

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

				//float rscaling = (1.0 / rij);
				//float drscaling = -1.0 / powf(rij, 2.0);

				float rscaling = (1.0 / powf(rij, 2.0 + gto_power));
				float drscaling = -(2.0 + float(gto_power)) * (1.0 / powf(rij, 3.0 + float(gto_power)));

				float ang = powf(rijx, gto_component_x) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);

				float dang[3];

				dang[0] = 0.0;
				dang[1] = 0.0;
				dang[2] = 0.0;

				//TODO need to make the following more efficient
				if (x == 0) {
					if (gto_component_x == 1.0) {
						dang[0] = -1.0 * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);
					} else if (gto_component_x > 1.0) {
						dang[0] = -gto_component_x * powf(rijx, gto_component_x - 1.0) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);
					}
				} else if (x == 1) {
					if (gto_component_y == 1.0) {
						dang[1] = -1.0 * powf(rijx, gto_component_x) * powf(rijz, gto_component_z);
					} else if (gto_component_y > 1.0) {
						dang[1] = -gto_component_y * powf(rijy, gto_component_y - 1.0) * powf(rijx, gto_component_x) * powf(rijz, gto_component_z);
					}
				} else {
					if (gto_component_z == 1.0) {
						dang[2] = -1.0 * powf(rijx, gto_component_x) * powf(rijy, gto_component_y);
					} else if (gto_component_z > 1.0) {
						dang[2] = -gto_component_z * powf(rijz, gto_component_z - 1.0) * powf(rijx, gto_component_x) * powf(rijy, gto_component_y);
					}
				}

				float drscalingx = drscaling * -drijx * ang * cut;
				float dangx = rscaling * dang[x] * cut;
				float dcutx = rscaling * ang * dcut * -drijx;

				float radial = expf(-eta * powf(rij - sgridpoints[z], 2.0));
				float dradial = -2.0 * eta * (rij - sgridpoints[z]);

				float val = sqrtf(eta / M_PI) * (1.0 / powf(rij, 2.0 + gto_power)) * ang * cut;

				float gval = radial * val;

				float dval_x = sqrtf_eta * radial * (drscalingx + dangx + dcutx + (rscaling * ang * cut * dradial * -drijx));

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

						atomicAdd(&lmax_deriv_temporary[lmn], sorbital_weights[korb] * 2.0 * val * val_deriv);
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
				grad[batchID][iatom][jatom][x][k] += lmax_deriv_temporary[k];
				grad[batchID][iatom][iatom][x][k] -= lmax_deriv_temporary[k];
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

				atomicAdd(&lmax_temporary[lmn], sorbital_weights[korb] * val * val);
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
		output[batchID][iatom][k] = lmax_temporary[k];
	}
}

/*

 __global__ void predict_forces_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
 const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
 const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
 const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> mbodylist,
 const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> gto_components,
 const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> orbital_weights,
 const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> gto_powers,
 const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> gridpoints, float eta, int lmax, float rcut,
 torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> alphas, torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> forces) {

 extern __shared__ int s[];

 int nbatch = coordinates.size(0);
 int ngauss = gridpoints.size(0);
 int natoms = coordinates.size(1);
 int norbs = gto_components.size(0);
 int nspecies = species.size(0);
 int nmbody = int(float((nspecies + 1) / 2) * nspecies);

 const int krepsize = nmbody * norbs * ngauss;
 const int lrepsize = nmbody * (lmax + 1) * ngauss;


 float *scoordinates_x = (float*) &s; //natoms
 float *scoordinates_y = (float*) &scoordinates_x[natoms]; //natoms
 float *scoordinates_z = (float*) &scoordinates_y[natoms]; //natoms
 int *selement_types = (int*) &scoordinates_z[natoms]; //natoms
 float *sgridpoints = (float*) &selement_types[natoms]; //ngaussians
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


 int batchID = int(floor(float(blockIdx.x / natoms)));
 int iatom = blockIdx.x % natoms;

 for (int jatom = threadIdx.x; jatom < natoms; jatom += blockDim.x) {
 scoordinates_x[jatom] = coordinates[batchID][jatom][0];
 scoordinates_y[jatom] = coordinates[batchID][jatom][1];
 scoordinates_z[jatom] = coordinates[batchID][jatom][2];
 selement_types[jatom] = element_types[batchID][jatom];
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

 float rix = scoordinates_x[iatom];
 float riy = scoordinates_y[iatom];
 float riz = scoordinates_z[iatom];

 double sqrtf_eta = sqrtf(eta / M_PI);

 __syncthreads();

 //need to generate the representation partially, first, for the derivative.
 for (int jatom = 0; jatom < natoms; jatom++) {

 if (iatom == jatom)
 continue;

 int element_type = selement_types[jatom];

 float rjx = scoordinates_x[jatom];
 float rjy = scoordinates_y[jatom];
 float rjz = scoordinates_z[jatom];

 float rijx = rix - rjx;
 float rijy = riy - rjy;
 float rijz = riz - rjz;

 float rij = sqrtf(rijx * rijx + rijy * rijy + rijz * rijz);

 if (rij > rcut) {
 continue;
 }

 float cut = 0.5 * (__cosf(rij * M_PI / rcut) + 1.0);

 for (int k = threadIdx.x; k < norbs * ngauss; k += blockDim.x) {

 int z = k % ngauss;
 int korb = int(floor(float(k / ngauss)));

 int gto_power = sgto_powers[korb];

 float gto_component_x = sgto_components_x[korb];
 float gto_component_y = sgto_components_y[korb];
 float gto_component_z = sgto_components_z[korb];

 float ang = powf(rijx, gto_component_x) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);

 float val = sqrtf_eta * (1.0 / powf(rij, 2.0 + gto_power)) * ang * cut;

 //float val = sqrtf_eta * (1.0 / rij) * ang * cut;

 float gval = expf(-eta * powf(rij - sgridpoints[z], 2.0)) * val;

 for (int m = 0; m < nspecies; m++) {

 int ej = smbodylist[element_type * nspecies + m];

 int kidx = korb * nmbody * ngauss + ej * ngauss + z;

 norb_temporary[kidx] += gval;
 }
 }
 }
 __syncthreads();

 float drij[3];
 //now lets do the derivative
 for (int jatom = 0; jatom < natoms; jatom++) {

 if (iatom == jatom)
 continue;

 int element_type = selement_types[jatom];

 float rjx = scoordinates_x[jatom];
 float rjy = scoordinates_y[jatom];
 float rjz = scoordinates_z[jatom];

 float rijx = rix - rjx;
 float rijy = riy - rjy;
 float rijz = riz - rjz;

 drij[0] = rijx;
 drij[1] = rijy;
 drij[2] = rijz;

 float rij = sqrtf(rijx * rijx + rijy * rijy + rijz * rijz);

 if (rij > rcut) {
 continue;
 }

 float cut = 0.5 * (__cosf(rij * M_PI / rcut) + 1.0);
 float dcut = -0.5 * (__sinf(rij * M_PI / rcut)) * M_PI / rcut;

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
 int korb = int(floor(float(k / ngauss)));

 int gto_power = sgto_powers[korb];

 float gto_component_x = sgto_components_x[korb];
 float gto_component_y = sgto_components_y[korb];
 float gto_component_z = sgto_components_z[korb];

 //float rscaling = (1.0 / rij);
 //float drscaling = -1.0 / powf(rij, 2.0);

 float rscaling = (1.0 / powf(rij, 2.0 + gto_power));
 float drscaling = -(2.0 + float(gto_power)) * (1.0 / powf(rij, 3.0 + float(gto_power)));

 float ang = powf(rijx, gto_component_x) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);

 float dang[3];

 dang[0] = 0.0;
 dang[1] = 0.0;
 dang[2] = 0.0;

 //TODO need to make the following more efficient
 if (x == 0) {
 if (gto_component_x == 1.0) {
 dang[0] = -1.0 * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);
 } else if (gto_component_x > 1.0) {
 dang[0] = -gto_component_x * powf(rijx, gto_component_x - 1.0) * powf(rijy, gto_component_y) * powf(rijz, gto_component_z);
 }
 } else if (x == 1) {
 if (gto_component_y == 1.0) {
 dang[1] = -1.0 * powf(rijx, gto_component_x) * powf(rijz, gto_component_z);
 } else if (gto_component_y > 1.0) {
 dang[1] = -gto_component_y * powf(rijy, gto_component_y - 1.0) * powf(rijx, gto_component_x) * powf(rijz, gto_component_z);
 }
 } else {
 if (gto_component_z == 1.0) {
 dang[2] = -1.0 * powf(rijx, gto_component_x) * powf(rijy, gto_component_y);
 } else if (gto_component_z > 1.0) {
 dang[2] = -gto_component_z * powf(rijz, gto_component_z - 1.0) * powf(rijx, gto_component_x) * powf(rijy, gto_component_y);
 }
 }

 float drscalingx = drscaling * -drijx * ang * cut;
 float dangx = rscaling * dang[x] * cut;
 float dcutx = rscaling * ang * dcut * -drijx;

 float radial = expf(-eta * powf(rij - sgridpoints[z], 2.0));
 float dradial = -2.0 * eta * (rij - sgridpoints[z]);

 float val = sqrtf(eta / M_PI) * (1.0 / powf(rij, 2.0 + gto_power)) * ang * cut;

 float gval = radial * val;

 float dval_x = sqrtf_eta * radial * (drscalingx + dangx + dcutx + (rscaling * ang * cut * dradial * -drijx));

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
 int korb = int(floor(float(k / ngauss)));

 int lchannel = sgto_powers[korb];

 for (int m = 0; m < nspecies; m++) {
 for (int n = m; n < nspecies; n++) {

 int mnidx = smbodylist[m * nspecies + n];

 int kmn = korb * nmbody * ngauss + mnidx * ngauss + z;
 int lmn = lchannel * nmbody * ngauss + mnidx * ngauss + z;

 float val = norb_temporary[kmn];
 float val_deriv = norb_deriv_temporary[kmn];

 atomicAdd(&lmax_deriv_temporary[lmn], sorbital_weights[korb] * 2.0 * val * val_deriv);
 }
 }
 }

 __syncthreads();

 //subtract single-element contributions
 for (int k = threadIdx.x; k < (lmax + 1) * ngauss; k += blockDim.x) {

 int z = k % ngauss;
 int l = int(floor(float(k / ngauss)));

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

 //TODO now project lmax_deriv_temporary  onto npcas in norb_deriv_temporary

 int tid = threadIdx.x + blockIdx.x * blockDim.x;

 for (int k = threadIdx.x; k < npcas; k += blockDim.x) {

 float sum = 0;

 for (int i = 0; i < lrepsize; i++)
 sum += vec[i] * reductor[i][k];

 norb_deriv_temporary[k] = sum;
 }

 //TODO now create the features from lmax_deriv_temporary

 for (int stack = 0; stack < nstacks; stack++) {

 //TODO now contract these sub-features /w sub-alphas

 }

 }
 }

 }*/

__global__
void get_element_types_kernel(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types) {

	int natoms = coordinates.size(1);
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

__global__
void get_element_bounds_kernel(const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_starts,
		torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nelements) {

	int nbatch = element_types.size(0);
	int natoms = element_types.size(1);
	int nspecies = species.size(0);

	extern __shared__ int s[];

	int *satomtypes = s;
	int *selement_starts = &s[blockDim.x];
	int *snelements = &selement_starts[nspecies];
	for (int iatom = threadIdx.x; iatom < natoms; iatom += blockDim.x) {

		int curr_type = element_types[blockIdx.x][iatom];

		satomtypes[iatom] = curr_type;

		__syncthreads();

		if (iatom + 1 < natoms && iatom + 1 < blockDim.x) {

			int next_type = satomtypes[iatom + 1];

			if (next_type != curr_type) {
				//found boundary
				selement_starts[next_type] = iatom + 1;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {

		int partial_sum = 0;

		for (int i = 0; i < nspecies - 1; i++) {
			int diff = selement_starts[i + 1] - selement_starts[i];
			snelements[i] = diff;
			partial_sum += diff;
		}

		snelements[nspecies - 1] = natoms - partial_sum;

//now copy to global memory

		for (int i = 0; i < nspecies; i++) {
			element_starts[blockIdx.x][i] = selement_starts[i];
			nelements[blockIdx.x][i] = snelements[i];
		}
	}
}

void getElementTypesGPU(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types) {

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);
	const int nthreads = 32;

	get_element_types_kernel<<<nbatch, nthreads>>>(coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void getElementTypeBoundsGPU(torch::Tensor element_types, torch::Tensor species, torch::Tensor element_starts, torch::Tensor nelements) {

	int nbatch = element_types.size(0);
	int natoms = element_types.size(1);
	int nspecies = species.size(0);

	const int nthreads = 32;

	get_element_bounds_kernel<<<nbatch, natoms, (nthreads + 2*nspecies)*sizeof(int)>>>(
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_starts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			nelements.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void elementalGTOGPUSharedMem_float(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor mbodylist, torch::Tensor gto_components, torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, float eta,
		int lmax, float rcut, torch::Tensor gto_output) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int nthreads = 64;

	int natoms = coordinates.size(1);
	int nbatch = coordinates.size(0);

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int(((float(nspecies) + 1.0) / 2.0) * nspecies);

	int currBatch = nbatch * natoms;

	int shared_mem_size = 5 * natoms + 2 * nspecies + ngaussians + 5 * norbs + (norbs * nmbody * ngaussians) + (lmax + 1) * nmbody * ngaussians;

	cudaEventRecord(start);
	EGTO_atomic_kernel_gridpara_float<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			mbodylist.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			gto_components.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			orbital_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			gto_powers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			gridpoints.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), eta, lmax,
			rcut, gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("elemental gto rep time: %f\n", milliseconds);
}

void elementalGTOGPUSharedMemDerivative_float(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor mbodylist, torch::Tensor gto_components, torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, float eta,
		int lmax, float rcut, torch::Tensor gto_output, torch::Tensor gto_output_derivative) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int nthreads = 64;

	int natoms = coordinates.size(1);
	int nbatch = coordinates.size(0);

	int ngaussians = gridpoints.size(0);
	int nspecies = species.size(0);
	int norbs = gto_components.size(0);

	int nmbody = int(((float(nspecies) + 1.0) / 2.0) * nspecies);

	int currBatch = nbatch * natoms;

	int shared_mem_size = 5 * natoms + 2 * nspecies + ngaussians + 5 * norbs + 2 * ((norbs * nmbody * ngaussians) + (lmax + 1) * nmbody * ngaussians);

	cudaEventRecord(start);
	EGTO_atomic_derivative_kernel_float<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			mbodylist.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			gto_components.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			orbital_weights.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			gto_powers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			gridpoints.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), eta, lmax,
			rcut, gto_output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			gto_output_derivative.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("ngto rep + derivative time: %f\n", milliseconds);
}

