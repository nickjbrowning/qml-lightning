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

__device__ void get_pbc_drij(float *drij, float *cell_vectors, float *inv_cell_vectors) {

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

__device__ float get_cutoff(float rij, float rcut, float rswitch, int cutoff_type) {
	float cut = 1.0;

	switch (cutoff_type) {

	case COSINE_CUTOFF:
		cut = 0.5 * (cosf(rij * M_PI / rcut) + 1.0);
		break;

	case SWITCH_FUNCTION:
		if (rij > rswitch) {
			float sx = (rij - rswitch) / (rcut - rswitch);
			cut = cut - 6.0 * powf(sx, 5.0) + 15.0 * powf(sx, 4.0) - 10.0 * powf(sx, 3.0);
		}
		break;
	default:
		cut = 0.5 * (cosf(rij * M_PI / rcut) + 1.0);
		break;
	}

	return cut;
}

__device__ float get_cutoff_derivative(float rij, float rcut, float rswitch, int cutoff_type) {
	float dcut = 0.0;

	switch (cutoff_type) {

	case COSINE_CUTOFF:

		dcut = -0.5 * (sinf(rij * M_PI / rcut)) * M_PI / rcut;
		break;

	case SWITCH_FUNCTION:

		if (rij > rswitch) {
			float sx = (rij - rswitch) / (rcut - rswitch);
			dcut = (1.0 / (rcut - rswitch)) * (-30.0 * powf(sx, 4.0) + 60.0 * powf(sx, 3.0) - 30.0 * powf(sx, 2.0));
		}
		break;

	default:
		dcut = -0.5 * (sinf(rij * M_PI / rcut)) * M_PI / rcut;
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
		d = sqrt(eta / M_PI) * expf(-eta * powf(rij - gridpoints[index], 2.0));
		break;

	case LOGNORMAL_DISTRIBUTION:
		mu = log(rij / sqrt(1.0 + eta / powf(rij, 2.0)));
		sigma2 = log(1.0 + eta / powf(rij, 2.0));
		sigma = sqrt(sigma2);

		d = 1.0 / (gridpoints[index] * sigma * SQRT2PI) * expf(-powf(log(gridpoints[index]) - mu, 2.0) / (2.0 * sigma2));
		break;

	case EXPEXP_DISTRIBUTION:
		d = expf(-eta * powf(expf(-rij) - gridpoints[index], 2.0));
		break;

	default:
		d = sqrt(eta / M_PI) * expf(-eta * powf(rij - gridpoints[index], 2.0));
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
		dradial_dx = sqrt_eta * expf(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		break;

	case LOGNORMAL_DISTRIBUTION:

		mu = log(rij / sqrt(1.0 + (eta / powf(rij, 2.0))));
		sigma = sqrt(log(1.0 + (eta / powf(rij, 2.0))));
		sigma2 = powf(sigma, 2.0);
		sigma4 = powf(sigma, 4.0);
		lnRs = log(gridpoints[index]);
		exp_ln = expf(-powf(lnRs - mu, 2.0) / powf(sigma, 2.0) * 0.5);
		rij2 = powf(rij, 2.0);

		dsigma_dx = drijx * eta * (1.0 / ((eta + rij2) * rij * sqrt(log((eta + rij2) / rij2))));
		dmu_dx = -drijx * ((2 * eta + rij2) / ((eta + rij2) * rij));

		dradial_dx = (sqrt(2.0) / (2 * sqrt(M_PI) * gridpoints[index] * sigma4))
				* (((mu - lnRs) * dsigma_dx - sigma * dmu_dx) * (mu - lnRs) - sigma2 * dsigma_dx) * exp_ln;

		break;

	case EXPEXP_DISTRIBUTION:
		dradial_dx = 2.0 * eta * (-gridpoints[index] + expf(-rij)) * expf(-eta * powf(expf(-rij) - gridpoints[index], 2.0)) * expf(-rij) * -drijx;
		break;

	default:
		dradial_dx = sqrt_eta * expf(-eta * powf(rij - gridpoints[index], 2.0)) * -eta * 2.0 * (rij - gridpoints[index]) * -drijx;
		break;
	}

	return dradial_dx;
}

__device__ float dot_abcd(float *ab, float *cd) {
	return ab[0] * cd[0] + ab[1] * cd[1] + ab[2] * cd[2];
}

__device__ float dot(float *v1, float *v2, float *v3, float *v4) {
	return (v1[0] - v2[0]) * (v3[0] - v4[0]) + (v1[1] - v2[1]) * (v3[1] - v4[1]) + (v1[2] - v2[2]) * (v3[2] - v4[2]);
}

//			float cos_1 = calc_cos_angle(rj, ri, rk); // ji, ki
//			float cos_2 = calc_cos_angle(rj, rk, ri); // jk, ik
//			float cos_3 = calc_cos_angle(ri, rj, rk); // ij, kj

__device__ float calc_cos_angle_abcb(float *ab, float *cb) {

	float v1norm = sqrt(ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
	float v2norm = sqrt(cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2]);

	float v1[3];
	float v2[3];

	v1[0] = ab[0] / v1norm;
	v1[1] = ab[1] / v1norm;
	v1[2] = ab[2] / v1norm;

	v2[0] = cb[0] / v2norm;
	v2[1] = cb[1] / v2norm;
	v2[2] = cb[2] / v2norm;

	float cos_angle = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; //v1.dot(v2);

	return cos_angle;
}

__device__ float calc_cos_angle(float *a, float *b, float *c) {

	float v1norm = sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
	float v2norm = sqrt((c[0] - b[0]) * (c[0] - b[0]) + (c[1] - b[1]) * (c[1] - b[1]) + (c[2] - b[2]) * (c[2] - b[2]));

	float v1[3];
	float v2[3];

	v1[0] = (a[0] - b[0]) / v1norm;
	v1[1] = (a[1] - b[1]) / v1norm;
	v1[2] = (a[2] - b[2]) / v1norm;

	v2[0] = (c[0] - b[0]) / v2norm;
	v2[1] = (c[1] - b[1]) / v2norm;
	v2[2] = (c[2] - b[2]) / v2norm;

	float cos_angle = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; //v1.dot(v2);

	return cos_angle;
}

__device__ float calc_angle(float *a, float *b, float *c) {

	float cos_angle = calc_cos_angle(a, b, c);

	if (cos_angle > 1.0)
		cos_angle = 1.0;
	if (cos_angle < -1.0)
		cos_angle = -1.0;

	return acosf(cos_angle);

}

__device__ float calc_angle_abcb(float *ab, float *cb) {

	float cos_angle = calc_cos_angle_abcb(ab, cb);

	if (cos_angle > 1.0)
		cos_angle = 1.0;
	if (cos_angle < -1.0)
		cos_angle = -1.0;

	return acosf(cos_angle);

}

__global__ void fchl19_representation_cuda_old(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs2,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);

	int nelements = species.size(0);

	float *scoords_x = (float*) &s;
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];

	float *sRs2 = (float*) &selement_types[max_neighbours];
	float *sRs3 = (float*) &sRs2[nRs2];

	float *scell = (float*) &sRs3[nRs3];
	float *sinv_cell = (float*) &scell[9];

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];

	int repsize = species.size(0) * nRs2 + (species.size(0) * (species.size(0) + 1)) * nRs3;

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs2; i += blockDim.x * blockDim.y) {
		sRs2[i] = Rs2[i];
	}
	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs3; i += blockDim.x * blockDim.y) {
		sRs3[i] = Rs3[i];
	}

	__syncthreads();

	int nneighbours_i = nneighbours[molID][iatom];

	for (int jatom = threadIdx.y * blockDim.x + threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x * blockDim.y) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];

	}
	__syncthreads();

	float ri[3];
	float rj[3];
	float rk[3];

	float drij[3];
	float drik[3];
	float drjk[3];
	float drji[3];
	float drki[3];
	float drkj[3];

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					scell[i * 3 + j] = cell[molID][i][j];
					sinv_cell[i * 3 + j] = inv_cell[molID][i][j];
				}
			}
		}
	}
	__syncthreads();

	ri[0] = coordinates[molID][iatom][0];
	ri[1] = coordinates[molID][iatom][1];
	ri[2] = coordinates[molID][iatom][2];

	float expf_v = expf(-powf(M_PI, 2) * 0.5);
	float sqrt2pi = sqrtf(2.0 * M_PI);

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		rj[0] = scoords_x[jatom];
		rj[1] = scoords_y[jatom];
		rj[2] = scoords_z[jatom];

		int jelement = selement_types[jatom];

		drij[0] = ri[0] - rj[0];
		drij[1] = ri[1] - rj[1];
		drij[2] = ri[2] - rj[2];

		if (pbc) {
			get_pbc_drij(drij, scell, sinv_cell);
		}

		drji[0] = -drij[0];
		drji[1] = -drij[1];
		drji[2] = -drij[2];

		float rij = sqrtf(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

		float scaling = 1.0 / powf(rij, two_body_decay);

		float rcutij = get_cutoff(rij, rcut, 0.0, 0);

		float mu = log(rij / sqrt(1.0 + eta2 / powf(rij, 2.0)));
		float sigma = sqrtf(log(1.0 + eta2 / powf(rij, 2.0)));

		float invsigma22 = 1.0 / (2.0 * powf(sigma, 2));

		for (int z = 0; z < nRs2; z++) {

			float radial = 1.0 / (sigma * sqrt2pi * sRs2[z]) * expf(-powf(log(sRs2[z]) - mu, 2) * invsigma22) * scaling * rcutij;

			atomicAdd(&output[molID][iatom][jelement * nRs2 + z], radial);

		}

		for (int katom = jatom + 1 + threadIdx.y; katom < nneighbours_i; katom += blockDim.y) {

			rk[0] = scoords_x[katom];
			rk[1] = scoords_y[katom];
			rk[2] = scoords_z[katom];

			int kelement = selement_types[katom];

			drik[0] = ri[0] - rk[0];
			drik[1] = ri[1] - rk[1];
			drik[2] = ri[2] - rk[2];

			if (pbc) {
				get_pbc_drij(drik, scell, sinv_cell);
			}
			drjk[0] = drik[0] - drij[0];
			drjk[1] = drik[1] - drij[1];
			drjk[2] = drik[2] - drij[2];

			drki[0] = -drik[0];
			drki[1] = -drik[1];
			drki[2] = -drik[2];

			drkj[0] = -drjk[0];
			drkj[1] = -drjk[1];
			drkj[2] = -drjk[2];

			float rik = sqrt(drik[0] * drik[0] + drik[1] * drik[1] + drik[2] * drik[2]);

			if (rik > rcut) {
				continue;
			}

			float rjk = sqrt(drjk[0] * drjk[0] + drjk[1] * drjk[1] + drjk[2] * drjk[2]);

			float rcutik = get_cutoff(rik, rcut, 0.0, 0);

			//__device__ float calc_cos_angle(float *a, float *b, float *c) {
			//float v1norm = sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
			//float v2norm = sqrt((c[0] - b[0]) * (c[0] - b[0]) + (c[1] - b[1]) * (c[1] - b[1]) + (c[2] - b[2]) * (c[2] - b[2]));

			//float angle = calc_angle(rj, ri, rk);
			float angle = calc_angle_abcb(drji, drki);

			//float cos_1 = calc_cos_angle(rj, ri, rk); // ji, ki
			float cos_1 = calc_cos_angle_abcb(drji, drki); // ji, ki
			//float cos_2 = calc_cos_angle(rj, rk, ri); // jk, ik
			float cos_2 = calc_cos_angle_abcb(drjk, drik); // jk, ik
			//float cos_3 = calc_cos_angle(ri, rj, rk); // ij, kj
			float cos_3 = calc_cos_angle_abcb(drij, drkj); // ij, kj

			float ksi3 = three_body_weight * (1.0 + 3 * cos_1 * cos_2 * cos_3) / powf(rij * rik * rjk, three_body_decay);

			float cos_angle = expf_v * 2.0 * cosf(angle);
			float sin_angle = expf_v * 2.0 * sinf(angle);

			int p = min(jelement, kelement);
			int q = max(jelement, kelement);

			int s = nelements * nRs2 + nRs3 * 2 * (-(p * (p + 1)) / 2 + q + nelements * p);

			for (int l = 0; l < nRs3; l++) {

				int z = s + l * 2;

				float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0)) * rcutik * rcutij;

				atomicAdd(&output[molID][iatom][z], radial * cos_angle * ksi3);
				atomicAdd(&output[molID][iatom][z + 1], radial * sin_angle * ksi3);
			}
		}
	}
}

__global__ void fchl19_representation_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs2,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);

	int nelements = species.size(0);

	float *scoords_x = (float*) &s;
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];

	float *sRs2 = (float*) &selement_types[max_neighbours];
	float *sRs3 = (float*) &sRs2[nRs2];

	float *scell = (float*) &sRs3[nRs3];
	float *sinv_cell = (float*) &scell[9];

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs2; i += blockDim.x * blockDim.y) {
		sRs2[i] = Rs2[i];
	}
	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs3; i += blockDim.x * blockDim.y) {
		sRs3[i] = Rs3[i];
	}

	__syncthreads();

	int nneighbours_i = nneighbours[molID][iatom];

	for (int jatom = threadIdx.y * blockDim.x + threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x * blockDim.y) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];

	}

	__syncthreads();

	float ri[3];

	float drij[3];
	float drik[3];
	float drjk[3];
	float drji[3];
	float drki[3];
	float drkj[3];

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					scell[i * 3 + j] = cell[molID][i][j];
					sinv_cell[i * 3 + j] = inv_cell[molID][i][j];
				}
			}
		}
	}
	__syncthreads();

	ri[0] = coordinates[molID][iatom][0];
	ri[1] = coordinates[molID][iatom][1];
	ri[2] = coordinates[molID][iatom][2];

	float expf_v = expf(-powf(M_PI, 2) * 0.5);
	float sqrt2pi = sqrtf(2.0 * M_PI);

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		int jelement = selement_types[jatom];

		drij[0] = ri[0] - scoords_x[jatom];
		drij[1] = ri[1] - scoords_y[jatom];
		drij[2] = ri[2] - scoords_z[jatom];

		if (pbc) {
			get_pbc_drij(drij, scell, sinv_cell);
		}

		drji[0] = -drij[0];
		drji[1] = -drij[1];
		drji[2] = -drij[2];

		float rij = sqrtf(drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2]);

		float scaling = 1.0 / powf(rij, two_body_decay);

		float rcutij = get_cutoff(rij, rcut, 0.0, 0);

		float mu = log(rij / sqrt(1.0 + eta2 / powf(rij, 2.0)));
		float sigma = sqrtf(log(1.0 + eta2 / powf(rij, 2.0)));

		float invsigma22 = 1.0 / (2.0 * powf(sigma, 2));

		for (int z = threadIdx.y; z < nRs2; z += blockDim.y) {

			float radial = 1.0 / (sigma * sqrt2pi * sRs2[z]) * expf(-powf(log(sRs2[z]) - mu, 2) * invsigma22) * scaling * rcutij;

			atomicAdd(&output[molID][iatom][jelement * nRs2 + z], radial);

		}

		for (int katom = jatom + 1 + threadIdx.y; katom < nneighbours_i; katom += blockDim.y) {

			int kelement = selement_types[katom];

			drik[0] = ri[0] - scoords_x[katom];
			drik[1] = ri[1] - scoords_y[katom];
			drik[2] = ri[2] - scoords_z[katom];

			if (pbc) {
				get_pbc_drij(drik, scell, sinv_cell);
			}

			drjk[0] = drik[0] - drij[0];
			drjk[1] = drik[1] - drij[1];
			drjk[2] = drik[2] - drij[2];

			drki[0] = -drik[0];
			drki[1] = -drik[1];
			drki[2] = -drik[2];

			drkj[0] = -drjk[0];
			drkj[1] = -drjk[1];
			drkj[2] = -drjk[2];

			float rik = sqrt(drik[0] * drik[0] + drik[1] * drik[1] + drik[2] * drik[2]);

			if (rik > rcut) {
				continue;
			}

			float rjk = sqrt(drjk[0] * drjk[0] + drjk[1] * drjk[1] + drjk[2] * drjk[2]);

			float rcutik = get_cutoff(rik, rcut, 0.0, 0);

			//__device__ float calc_cos_angle(float *a, float *b, float *c) {
			//float v1norm = sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
			//float v2norm = sqrt((c[0] - b[0]) * (c[0] - b[0]) + (c[1] - b[1]) * (c[1] - b[1]) + (c[2] - b[2]) * (c[2] - b[2]));

			//float angle = calc_angle(rj, ri, rk);
			float angle = calc_angle_abcb(drji, drki);

			//float cos_1 = calc_cos_angle(rj, ri, rk); // ji, ki
			float cos_1 = calc_cos_angle_abcb(drji, drki); // ji, ki
			//float cos_2 = calc_cos_angle(rj, rk, ri); // jk, ik
			float cos_2 = calc_cos_angle_abcb(drjk, drik); // jk, ik
			//float cos_3 = calc_cos_angle(ri, rj, rk); // ij, kj
			float cos_3 = calc_cos_angle_abcb(drij, drkj); // ij, kj

			float ksi3 = three_body_weight * (1.0 + 3 * cos_1 * cos_2 * cos_3) / powf(rij * rik * rjk, three_body_decay);

			float cos_angle = expf_v * 2.0 * cosf(angle);
			float sin_angle = expf_v * 2.0 * sinf(angle);

			int p = min(jelement, kelement);
			int q = max(jelement, kelement);

			int s = nelements * nRs2 + nRs3 * 2 * (-(p * (p + 1)) / 2 + q + nelements * p);

			for (int l = 0; l < nRs3; l++) {

				int z = s + l * 2;

				float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0)) * rcutik * rcutij;

				atomicAdd(&output[molID][iatom][z], radial * cos_angle * ksi3);
				atomicAdd(&output[molID][iatom][z + 1], radial * sin_angle * ksi3);
			}
		}
	}
}

__global__ void fchl19_representation_and_derivative_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs,
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs2,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output,
		torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> grad) {

	extern __shared__ int s[];

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);

	int nelements = species.size(0);

	float *scoords_x = (float*) &s;
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];
	int *sneighbours = (int*) &selement_types[max_neighbours];

	float *sRs2 = (float*) &sneighbours[max_neighbours];
	float *sRs3 = (float*) &sRs2[nRs2];

	float *scell = (float*) &sRs3[nRs3];
	float *sinv_cell = (float*) &scell[9];

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];
		sneighbours[jatom] = j;

	}

	for (int i = threadIdx.x; i < nRs2; i += blockDim.x) {
		sRs2[i] = Rs2[i];
	}

	for (int i = threadIdx.x; i < nRs3; i += blockDim.x) {
		sRs3[i] = Rs3[i];
	}

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					scell[i * 3 + j] = cell[molID][i][j];
					sinv_cell[i * 3 + j] = inv_cell[molID][i][j];
				}
			}
		}
	}

	__syncthreads();

	float ri[3];
	float rj[3];
	float rk[3];

	float drij[3];
	float drik[3];
	float drjk[3];

	ri[0] = coordinates[molID][iatom][0];
	ri[1] = coordinates[molID][iatom][1];
	ri[2] = coordinates[molID][iatom][2];

	float ielement = element_types[molID][iatom];

	float invcut = 1.0 / rcut;

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		rj[0] = scoords_x[jatom];
		rj[1] = scoords_y[jatom];
		rj[2] = scoords_z[jatom];

		int j = sneighbours[jatom];
		int jelement = selement_types[jatom];

		drij[0] = ri[0] - rj[0];
		drij[1] = ri[1] - rj[1];
		drij[2] = ri[2] - rj[2];

		if (pbc) {
			get_pbc_drij(drij, scell, sinv_cell);
		}

		float drji[3] = { -drij[0], -drij[1], -drij[2] };

		float rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];
		float rij = sqrtf(rij2);
		float invrij = 1.0 / rij;
		float invrij2 = invrij * invrij;

		float scaling = 1.0 / powf(rij, two_body_decay);

		float rcutij = get_cutoff(rij, rcut, 0.0, 0);

		float mu = log(rij / sqrt(1.0 + eta2 / powf(rij, 2.0)));
		float sigma = sqrt(log(1.0 + eta2 / powf(rij, 2.0)));

		float dcut = get_cutoff_derivative(rij, rcut, 0.0, 0);

		float dscal = -two_body_decay / powf(rij, two_body_decay + 1.0);

		for (int z = 0; z < nRs2; z++) {

			float radial = 1.0 / (sigma * sqrt(2.0 * M_PI) * sRs2[z]) * expf(-powf(log(sRs2[z]) - mu, 2) / (2.0 * powf(sigma, 2)));

			float rep = radial * scaling * rcutij;

			atomicAdd(&output[molID][iatom][jelement * nRs2 + z], rep);

			for (int x = 0; x < 3; x++) {

				float dx = drij[x] / rij;

				float dradialx = get_radial_derivative_distribution(dx, rij, eta2, sRs2, z, 1);

				float dcutx = dcut * -dx;

				float dscalingx = dscal * -dx;

				float deriv = dradialx * scaling * rcutij + radial * dscalingx * rcutij + radial * scaling * dcutx;

				atomicAdd(&grad[molID][iatom][iatom][x][jelement * nRs2 + z], -deriv);
				atomicAdd(&grad[molID][iatom][j][x][jelement * nRs2 + z], deriv);

			}

		}

		for (int katom = jatom + 1; katom < nneighbours_i; katom++) {

			rk[0] = scoords_x[katom];
			rk[1] = scoords_y[katom];
			rk[2] = scoords_z[katom];

			int kelement = selement_types[katom];
			int k = sneighbours[katom];

			drik[0] = ri[0] - rk[0];
			drik[1] = ri[1] - rk[1];
			drik[2] = ri[2] - rk[2];

			if (pbc) {
				get_pbc_drij(drik, scell, sinv_cell);
			}
			drjk[0] = drik[0] - drij[0];
			drjk[1] = drik[1] - drij[1];
			drjk[2] = drik[2] - drij[2];

			float drki[3] = { -drik[0], -drik[1], -drik[2] };
			float drkj[3] = { -drjk[0], -drjk[1], -drjk[2] };

			float rik2 = drik[0] * drik[0] + drik[1] * drik[1] + drik[2] * drik[2];
			float rik = sqrtf(rik2);

			float invrik = 1.0 / rik;
			float invrik2 = invrik * invrik;

			if (rik > rcut) {
				continue;
			}

			float rjk2 = drjk[0] * drjk[0] + drjk[1] * drjk[1] + drjk[2] * drjk[2];
			float rjk = sqrtf(rjk2);

			float invrjk = 1.0 / rjk;
			float invrjk2 = invrjk * invrjk;

			float rcutik = get_cutoff(rik, rcut, 0.0, 0);

			//float angle = calc_angle(rj, ri, rk);
			float angle = calc_angle_abcb(drji, drki);

			//float cos_i = calc_cos_angle(rj, ri, rk); // ji, ki
			float cos_i = calc_cos_angle_abcb(drji, drki); // ji, ki
			//float cos_k = calc_cos_angle(rj, rk, ri); // jk, ik
			float cos_k = calc_cos_angle_abcb(drjk, drik); // jk, ik
			//float cos_j = calc_cos_angle(ri, rj, rk); // ij, kj
			float cos_j = calc_cos_angle_abcb(drij, drkj); // ij, kj

			float cos_angle = expf(-powf(M_PI, 2) * 0.5) * 2.0 * cosf(angle);
			float sin_angle = expf(-powf(M_PI, 2) * 0.5) * 2.0 * sinf(angle);

			float invr_atm = powf(invrij * invrjk * invrik, three_body_decay);

			float atm = (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

			int p = min(jelement, kelement);
			int q = max(jelement, kelement);

			int s = nelements * nRs2 + nRs3 * 2 * (-(p * (p + 1)) / 2 + q + nelements * p);

			for (int l = 0; l < nRs3; l++) {

				int z = s + l * 2;

				float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0)) * rcutik * rcutij;

				atomicAdd(&output[molID][iatom][z], radial * cos_angle * atm);
				atomicAdd(&output[molID][iatom][z + 1], radial * sin_angle * atm);

			}

			//float vi = dot(rj, ri, rk, ri); // ji, ki
			float vi = dot_abcd(drji, drki);
			//float vj = dot(rk, rj, ri, rj); // kj, ij
			float vj = dot_abcd(drkj, drij);
			//float vk = dot(ri, rk, rj, rk); // ik, jk
			float vk = dot_abcd(drik, drjk);

			float dcos_angle = expf(-powf(M_PI, 2) * 0.5) * 2 * sinf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));
			float dsin_angle = -expf(-powf(M_PI, 2) * 0.5) * 2 * cosf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));

			float atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
			float atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
			float atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

			for (int x = 0; x < 3; x++) {

				float a = drji[x];
				float b = 0.0;
				float c = drki[x];

				float d_radial_d_j = (b - a) * invrij; // drij

				float d_radial_d_k = (b - c) * invrik;  // drik

				float d_radial_d_i = -(d_radial_d_j + d_radial_d_k);

				float d_angular_d_j = (c - b) + vi * ((b - a) * invrij2); // drki, drij

				float d_angular_d_k = (a - b) + vi * ((b - c) * invrik2); // drji, drik

				float d_angular_d_i = -(d_angular_d_j + d_angular_d_k);

				float d_ijdecay = -M_PI * (b - a) * sinf(M_PI * rij * invcut) * 0.5 * invrij * invcut; // drij

				float d_ikdecay = -M_PI * (b - c) * sinf(M_PI * rik * invcut) * 0.5 * invrik * invcut; // drik

				float d_atm_ii = 2 * b - a - c - vi * ((b - a) * invrij2 + (b - c) * invrik2); //- a - c = -drjk, drij, drik

				float d_atm_ij = c - a - vj * (b - a) * invrij2;				//drkj, drij

				float d_atm_ik = a - c - vk * (b - c) * invrik2;				//drjk, drik

				float d_atm_ji = c - b - vi * (a - b) * invrij2;				//drki, drji

				float d_atm_jj = 2 * a - b - c - vj * ((a - b) * invrij2 + (a - c) * invrjk2);

				float d_atm_jk = b - c - vk * (a - c) * invrjk2;

				float d_atm_ki = a - b - vi * (c - b) * invrik2;

				float d_atm_kj = b - a - vj * (c - a) * invrjk2;

				float d_atm_kk = 2 * c - a - b - vk * ((c - a) * invrjk2 + (c - b) * invrik2);

				float d_atm_extra_i = ((a - b) * invrij2 + (c - b) * invrik2) * atm * three_body_decay / three_body_weight;

				float d_atm_extra_j = ((b - a) * invrij2 + (c - a) * invrjk2) * atm * three_body_decay / three_body_weight;

				float d_atm_extra_k = ((a - c) * invrjk2 + (b - c) * invrik2) * atm * three_body_decay / three_body_weight;

				for (int l = 0; l < nRs3; l++) {

					float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0));
					float d_radial = radial * eta3 * (0.5 * (rij + rik) - sRs3[l]);

					int z = s + l * 2;

					atomicAdd(&grad[molID][iatom][iatom][x][z],
							dcos_angle * d_angular_d_i * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_i * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ii + atm_j * d_atm_ij + atm_k * d_atm_ik + d_atm_extra_i) * three_body_weight * rcutij
											* rcutik + cos_angle * radial * (d_ijdecay * rcutik + rcutij * d_ikdecay) * atm);

					atomicAdd(&grad[molID][iatom][iatom][x][z + 1],
							dsin_angle * d_angular_d_i * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_i * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ii + atm_j * d_atm_ij + atm_k * d_atm_ik + d_atm_extra_i) * three_body_weight * rcutij
											* rcutik + sin_angle * radial * (d_ijdecay * rcutik + rcutij * d_ikdecay) * atm);

					atomicAdd(&grad[molID][iatom][j][x][z],
							dcos_angle * d_angular_d_j * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_j * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ji + atm_j * d_atm_jj + atm_k * d_atm_jk + d_atm_extra_j) * three_body_weight * rcutij
											* rcutik - cos_angle * radial * d_ijdecay * rcutik * atm);

					atomicAdd(&grad[molID][iatom][j][x][z + 1],
							dsin_angle * d_angular_d_j * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_j * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ji + atm_j * d_atm_jj + atm_k * d_atm_jk + d_atm_extra_j) * three_body_weight * rcutij
											* rcutik - sin_angle * radial * d_ijdecay * rcutik * atm);

					atomicAdd(&grad[molID][iatom][k][x][z],
							dcos_angle * d_angular_d_k * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_k * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ki + atm_j * d_atm_kj + atm_k * d_atm_kk + d_atm_extra_k) * three_body_weight * rcutij
											* rcutik - cos_angle * radial * rcutij * d_ikdecay * atm);

					atomicAdd(&grad[molID][iatom][k][x][z + 1],
							dsin_angle * d_angular_d_k * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_k * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ki + atm_j * d_atm_kj + atm_k * d_atm_kk + d_atm_extra_k) * three_body_weight * rcutij
											* rcutik - sin_angle * radial * rcutij * d_ikdecay * atm);

				}
			}
		}
	}
}

__global__ void fchl19_derivative_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs2,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> grad) {

	extern __shared__ int s[];

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);

	int nelements = species.size(0);

	float *scoords_x = (float*) &s;
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];
	int *sneighbours = (int*) &selement_types[max_neighbours];

	float *sRs2 = (float*) &sneighbours[max_neighbours];
	float *sRs3 = (float*) &sRs2[nRs2];

	float *scell = (float*) &sRs3[nRs3];
	float *sinv_cell = (float*) &scell[9];

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	bool pbc = false;

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];
		sneighbours[jatom] = j;

	}

	for (int i = threadIdx.x; i < nRs2; i += blockDim.x) {
		sRs2[i] = Rs2[i];
	}

	for (int i = threadIdx.x; i < nRs3; i += blockDim.x) {
		sRs3[i] = Rs3[i];
	}

	__syncthreads();

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					scell[i * 3 + j] = cell[molID][i][j];
					sinv_cell[i * 3 + j] = inv_cell[molID][i][j];
				}
			}
		}
	}

	__syncthreads();

	float ri[3];
	float rj[3];
	float rk[3];

	float drij[3];
	float drik[3];
	float drjk[3];

	ri[0] = coordinates[molID][iatom][0];
	ri[1] = coordinates[molID][iatom][1];
	ri[2] = coordinates[molID][iatom][2];

	float ielement = element_types[molID][iatom];

	float invcut = 1.0 / rcut;

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		rj[0] = scoords_x[jatom];
		rj[1] = scoords_y[jatom];
		rj[2] = scoords_z[jatom];

		int j = sneighbours[jatom];
		int jelement = selement_types[jatom];

		drij[0] = ri[0] - rj[0];
		drij[1] = ri[1] - rj[1];
		drij[2] = ri[2] - rj[2];

		if (pbc) {
			get_pbc_drij(drij, scell, sinv_cell);
		}

		float drji[3] = { -drij[0], -drij[1], -drij[2] };

		float rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];
		float rij = sqrtf(rij2);
		float invrij = 1.0 / rij;
		float invrij2 = invrij * invrij;

		float scaling = 1.0 / powf(rij, two_body_decay);

		float rcutij = get_cutoff(rij, rcut, 0.0, 0);

		float mu = log(rij / sqrt(1.0 + eta2 / powf(rij, 2.0)));
		float sigma = sqrt(log(1.0 + eta2 / powf(rij, 2.0)));

		float dcut = get_cutoff_derivative(rij, rcut, 0.0, 0);

		float dscal = -two_body_decay / powf(rij, two_body_decay + 1.0);

		for (int z = 0; z < nRs2; z++) {

			float radial = 1.0 / (sigma * sqrt(2.0 * M_PI) * sRs2[z]) * expf(-powf(log(sRs2[z]) - mu, 2) / (2.0 * powf(sigma, 2)));

			for (int x = 0; x < 3; x++) {

				float dx = drij[x] / rij;

				float dradialx = get_radial_derivative_distribution(dx, rij, eta2, sRs2, z, 1);

				float dcutx = dcut * -dx;

				float dscalingx = dscal * -dx;

				float deriv = dradialx * scaling * rcutij + radial * dscalingx * rcutij + radial * scaling * dcutx;

				atomicAdd(&grad[molID][iatom][iatom][x][jelement * nRs2 + z], -deriv);
				atomicAdd(&grad[molID][iatom][j][x][jelement * nRs2 + z], deriv);

			}

		}

		for (int katom = jatom + 1; katom < nneighbours_i; katom++) {

			rk[0] = scoords_x[katom];
			rk[1] = scoords_y[katom];
			rk[2] = scoords_z[katom];

			int kelement = selement_types[katom];
			int k = sneighbours[katom];

			drik[0] = ri[0] - rk[0];
			drik[1] = ri[1] - rk[1];
			drik[2] = ri[2] - rk[2];

			if (pbc) {
				get_pbc_drij(drik, scell, sinv_cell);
			}

			drjk[0] = drik[0] - drij[0];
			drjk[1] = drik[1] - drij[1];
			drjk[2] = drik[2] - drij[2];

			float rik2 = drik[0] * drik[0] + drik[1] * drik[1] + drik[2] * drik[2];
			float rik = sqrtf(rik2);

			float invrik = 1.0 / rik;
			float invrik2 = invrik * invrik;

			if (rik > rcut) {
				continue;
			}

			float rjk2 = drjk[0] * drjk[0] + drjk[1] * drjk[1] + drjk[2] * drjk[2];
			float rjk = sqrtf(rjk2);

			float invrjk = 1.0 / rjk;
			float invrjk2 = invrjk * invrjk;

			float rcutik = get_cutoff(rik, rcut, 0.0, 0);

			float drki[3] = { -drik[0], -drik[1], -drik[2] };
			float drkj[3] = { -drjk[0], -drjk[1], -drjk[2] };

			//float angle = calc_angle(rj, ri, rk);
			float angle = calc_angle_abcb(drji, drki);

			//float cos_i = calc_cos_angle(rj, ri, rk); // ji, ki
			float cos_i = calc_cos_angle_abcb(drji, drki); // ji, ki
			//float cos_k = calc_cos_angle(rj, rk, ri); // jk, ik
			float cos_k = calc_cos_angle_abcb(drjk, drik); // jk, ik
			//float cos_j = calc_cos_angle(ri, rj, rk); // ij, kj
			float cos_j = calc_cos_angle_abcb(drij, drkj); // ij, kj

			float cos_angle = expf(-powf(M_PI, 2) * 0.5) * 2.0 * cosf(angle);
			float sin_angle = expf(-powf(M_PI, 2) * 0.5) * 2.0 * sinf(angle);

			float invr_atm = powf(invrij * invrjk * invrik, three_body_decay);

			float atm = (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

			int p = min(jelement, kelement);
			int q = max(jelement, kelement);

			int s = nelements * nRs2 + nRs3 * 2 * (-(p * (p + 1)) / 2 + q + nelements * p);

			//float vi = dot(rj, ri, rk, ri); // ji, ki
			float vi = dot_abcd(drji, drki);
			//float vj = dot(rk, rj, ri, rj); // kj, ij
			float vj = dot_abcd(drkj, drij);
			//float vk = dot(ri, rk, rj, rk); // ik, jk
			float vk = dot_abcd(drik, drjk);

			float dcos_angle = expf(-powf(M_PI, 2) * 0.5) * 2 * sinf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));
			float dsin_angle = -expf(-powf(M_PI, 2) * 0.5) * 2 * cosf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));

			float atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
			float atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
			float atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

			for (int x = 0; x < 3; x++) {

				float a = drji[x];
				float b = 0.0;
				float c = drki[x];

				float d_radial_d_j = (b - a) * invrij; // drij

				float d_radial_d_k = (b - c) * invrik;  // drik

				float d_radial_d_i = -(d_radial_d_j + d_radial_d_k);

				float d_angular_d_j = (c - b) + vi * ((b - a) * invrij2); // drki, drij

				float d_angular_d_k = (a - b) + vi * ((b - c) * invrik2); // drji, drik

				float d_angular_d_i = -(d_angular_d_j + d_angular_d_k);

				float d_ijdecay = -M_PI * (b - a) * sinf(M_PI * rij * invcut) * 0.5 * invrij * invcut; // drij

				float d_ikdecay = -M_PI * (b - c) * sinf(M_PI * rik * invcut) * 0.5 * invrik * invcut; // drik

				float d_atm_ii = 2 * b - a - c - vi * ((b - a) * invrij2 + (b - c) * invrik2); //- a - c = -drjk, drij, drik

				float d_atm_ij = c - a - vj * (b - a) * invrij2;				//drkj, drij

				float d_atm_ik = a - c - vk * (b - c) * invrik2;				//drjk, drik

				float d_atm_ji = c - b - vi * (a - b) * invrij2;				//drki, drji

				float d_atm_jj = 2 * a - b - c - vj * ((a - b) * invrij2 + (a - c) * invrjk2);

				float d_atm_jk = b - c - vk * (a - c) * invrjk2;

				float d_atm_ki = a - b - vi * (c - b) * invrik2;

				float d_atm_kj = b - a - vj * (c - a) * invrjk2;

				float d_atm_kk = 2 * c - a - b - vk * ((c - a) * invrjk2 + (c - b) * invrik2);

				float d_atm_extra_i = ((a - b) * invrij2 + (c - b) * invrik2) * atm * three_body_decay / three_body_weight;

				float d_atm_extra_j = ((b - a) * invrij2 + (c - a) * invrjk2) * atm * three_body_decay / three_body_weight;

				float d_atm_extra_k = ((a - c) * invrjk2 + (b - c) * invrik2) * atm * three_body_decay / three_body_weight;

				for (int l = 0; l < nRs3; l++) {

					float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0));
					float d_radial = radial * eta3 * (0.5 * (rij + rik) - sRs3[l]);

					int z = s + l * 2;

					atomicAdd(&grad[molID][iatom][iatom][x][z],
							dcos_angle * d_angular_d_i * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_i * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ii + atm_j * d_atm_ij + atm_k * d_atm_ik + d_atm_extra_i) * three_body_weight * rcutij
											* rcutik + cos_angle * radial * (d_ijdecay * rcutik + rcutij * d_ikdecay) * atm);

					atomicAdd(&grad[molID][iatom][iatom][x][z + 1],
							dsin_angle * d_angular_d_i * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_i * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ii + atm_j * d_atm_ij + atm_k * d_atm_ik + d_atm_extra_i) * three_body_weight * rcutij
											* rcutik + sin_angle * radial * (d_ijdecay * rcutik + rcutij * d_ikdecay) * atm);

					atomicAdd(&grad[molID][iatom][j][x][z],
							dcos_angle * d_angular_d_j * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_j * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ji + atm_j * d_atm_jj + atm_k * d_atm_jk + d_atm_extra_j) * three_body_weight * rcutij
											* rcutik - cos_angle * radial * d_ijdecay * rcutik * atm);

					atomicAdd(&grad[molID][iatom][j][x][z + 1],
							dsin_angle * d_angular_d_j * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_j * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ji + atm_j * d_atm_jj + atm_k * d_atm_jk + d_atm_extra_j) * three_body_weight * rcutij
											* rcutik - sin_angle * radial * d_ijdecay * rcutik * atm);

					atomicAdd(&grad[molID][iatom][k][x][z],
							dcos_angle * d_angular_d_k * radial * atm * rcutij * rcutik + cos_angle * d_radial * d_radial_d_k * atm * rcutij * rcutik
									+ cos_angle * radial * (atm_i * d_atm_ki + atm_j * d_atm_kj + atm_k * d_atm_kk + d_atm_extra_k) * three_body_weight * rcutij
											* rcutik - cos_angle * radial * rcutij * d_ikdecay * atm);

					atomicAdd(&grad[molID][iatom][k][x][z + 1],
							dsin_angle * d_angular_d_k * radial * atm * rcutij * rcutik + sin_angle * d_radial * d_radial_d_k * atm * rcutij * rcutik
									+ sin_angle * radial * (atm_i * d_atm_ki + atm_j * d_atm_kj + atm_k * d_atm_kk + d_atm_extra_k) * three_body_weight * rcutij
											* rcutik - sin_angle * radial * rcutij * d_ikdecay * atm);

				}
			}
		}
	}
}

__global__ void fchl19_backwards_cuda(const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> coordinates,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> charges,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> species,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> element_types,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> cell,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> inv_cell,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockAtomIDs, // blockIdx -> atom idx
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> blockMolIDs, // blockIdx -> molecule jdx
		const torch::PackedTensorAccessor32<int, 3, torch::RestrictPtrTraits> neighbourlist,
		const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> nneighbours, const int max_neighbours,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs2,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_in,
		torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> grad_out) {

	extern __shared__ int s[];

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);

	int nelements = species.size(0);

	float *scoords_x = (float*) &s;
	float *scoords_y = (float*) &scoords_x[max_neighbours];
	float *scoords_z = (float*) &scoords_y[max_neighbours];
	int *selement_types = (int*) &scoords_z[max_neighbours];
	int *sneighbours = (int*) &selement_types[max_neighbours];

	float *sRs2 = (float*) &sneighbours[max_neighbours];
	float *sRs3 = (float*) &sRs2[nRs2];

	float *scell = (float*) &sRs3[nRs3];
	float *sinv_cell = (float*) &scell[9];

	float *sgradx = (float*) &sinv_cell[9];
	float *sgrady = (float*) &sgradx[max_neighbours];
	float *sgradz = (float*) &sgrady[max_neighbours];

	int molID = blockMolIDs[blockIdx.x];
	int iatom = blockAtomIDs[blockIdx.x];
	int nneighbours_i = nneighbours[molID][iatom];

	for (int jatom = threadIdx.y * blockDim.x + threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x * blockDim.y) {

		int j = neighbourlist[molID][iatom][jatom];

		scoords_x[jatom] = coordinates[molID][j][0];
		scoords_y[jatom] = coordinates[molID][j][1];
		scoords_z[jatom] = coordinates[molID][j][2];
		selement_types[jatom] = element_types[molID][j];
		sneighbours[jatom] = j;

	}
	__syncthreads();

	float igrad[3] = { 0.0, 0.0, 0.0 };

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < max_neighbours; i += blockDim.x * blockDim.y) {
		sgradx[i] = 0.0;
		sgrady[i] = 0.0;
		sgradz[i] = 0.0;
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs2; i += blockDim.x * blockDim.y) {
		sRs2[i] = Rs2[i];
	}

	__syncthreads();

	for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < nRs3; i += blockDim.x * blockDim.y) {
		sRs3[i] = Rs3[i];
	}

	__syncthreads();

	bool pbc = false;

	if (cell.size(0) > 0) {

		pbc = true;

		if (threadIdx.x == 0 && threadIdx.y == 0) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					scell[i * 3 + j] = cell[molID][i][j];
					sinv_cell[i * 3 + j] = inv_cell[molID][i][j];
				}
			}
		}
	}

	__syncthreads();

	float ri[3];
	float rj[3];
	float rk[3];

	float drij[3];
	float drik[3];
	float drjk[3];

	ri[0] = coordinates[molID][iatom][0];
	ri[1] = coordinates[molID][iatom][1];
	ri[2] = coordinates[molID][iatom][2];

	float ielement = element_types[molID][iatom];

	float invcut = 1.0 / rcut;

	float zeta_factor = expf(-powf(M_PI, 2) * 0.5);

	for (int jatom = threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x) {

		float deriv_jatom[3] = { 0.0, 0.0, 0.0 };

		rj[0] = scoords_x[jatom];
		rj[1] = scoords_y[jatom];
		rj[2] = scoords_z[jatom];

		int jelement = selement_types[jatom];

		drij[0] = ri[0] - rj[0];
		drij[1] = ri[1] - rj[1];
		drij[2] = ri[2] - rj[2];

		if (pbc) {
			get_pbc_drij(drij, scell, sinv_cell);
		}

		float drji[3] = { -drij[0], -drij[1], -drij[2] };

		float rij2 = drij[0] * drij[0] + drij[1] * drij[1] + drij[2] * drij[2];
		float rij = sqrtf(rij2);
		float invrij = 1.0 / rij;
		float invrij2 = invrij * invrij;

		float scaling = 1.0 / powf(rij, two_body_decay);

		float rcutij = get_cutoff(rij, rcut, 0.0, 0);

		float mu = log(rij / sqrt(1.0 + eta2 / powf(rij, 2.0)));
		float sigma = sqrt(log(1.0 + eta2 / powf(rij, 2.0)));

		float dcut = get_cutoff_derivative(rij, rcut, 0.0, 0);

		float dscal = -two_body_decay / powf(rij, two_body_decay + 1.0);

		for (int z = threadIdx.y; z < nRs2; z += blockDim.y) {

			int idx = jelement * nRs2 + z;

			float radial = 1.0 / (sigma * sqrt(2.0 * M_PI) * sRs2[z]) * expf(-powf(log(sRs2[z]) - mu, 2) / (2.0 * powf(sigma, 2)));

			float grad_in_iatom = grad_in[molID][iatom][idx];

			float deriv[3];

			for (int x = 0; x < 3; x++) {

				float dx = drij[x] / rij;

				float dradialx = get_radial_derivative_distribution(dx, rij, eta2, sRs2, z, 1);

				float dcutx = dcut * -dx;

				float dscalingx = dscal * -dx;

				deriv[x] = dradialx * scaling * rcutij + radial * dscalingx * rcutij + radial * scaling * dcutx;

			}

			igrad[0] += grad_in_iatom * -deriv[0];
			igrad[1] += grad_in_iatom * -deriv[1];
			igrad[2] += grad_in_iatom * -deriv[2];

			deriv_jatom[0] += grad_in_iatom * deriv[0];
			deriv_jatom[1] += grad_in_iatom * deriv[1];
			deriv_jatom[2] += grad_in_iatom * deriv[2];

		}

		for (int katom = jatom + 1 + threadIdx.y; katom < nneighbours_i; katom += blockDim.y) {

			rk[0] = scoords_x[katom];
			rk[1] = scoords_y[katom];
			rk[2] = scoords_z[katom];

			int kelement = selement_types[katom];

			drik[0] = ri[0] - rk[0];
			drik[1] = ri[1] - rk[1];
			drik[2] = ri[2] - rk[2];

			if (pbc) {
				get_pbc_drij(drik, scell, sinv_cell);
			}

			drjk[0] = drik[0] - drij[0];
			drjk[1] = drik[1] - drij[1];
			drjk[2] = drik[2] - drij[2];

			float rik2 = drik[0] * drik[0] + drik[1] * drik[1] + drik[2] * drik[2];
			float rik = sqrtf(rik2);

			float invrik = 1.0 / rik;
			float invrik2 = invrik * invrik;

			if (rik > rcut) {
				continue;
			}

			float rjk2 = drjk[0] * drjk[0] + drjk[1] * drjk[1] + drjk[2] * drjk[2];
			float rjk = sqrtf(rjk2);

			float invrjk = 1.0 / rjk;
			float invrjk2 = invrjk * invrjk;

			float rcutik = get_cutoff(rik, rcut, 0.0, 0);

			float drki[3] = { -drik[0], -drik[1], -drik[2] };
			float drkj[3] = { -drjk[0], -drjk[1], -drjk[2] };

			//float angle = calc_angle(rj, ri, rk);
			float angle = calc_angle_abcb(drji, drki);

			//float cos_i = calc_cos_angle(rj, ri, rk); // ji, ki
			float cos_i = calc_cos_angle_abcb(drji, drki); // ji, ki
			//float cos_k = calc_cos_angle(rj, rk, ri); // jk, ik
			float cos_k = calc_cos_angle_abcb(drjk, drik); // jk, ik
			//float cos_j = calc_cos_angle(ri, rj, rk); // ij, kj
			float cos_j = calc_cos_angle_abcb(drij, drkj); // ij, kj

			float cos_angle = zeta_factor * 2.0 * cosf(angle);
			float sin_angle = zeta_factor * 2.0 * sinf(angle);

			float invr_atm = powf(invrij * invrjk * invrik, three_body_decay);

			float atm = (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

			int p = min(jelement, kelement);
			int q = max(jelement, kelement);

			int s = nelements * nRs2 + nRs3 * 2 * (-(p * (p + 1)) / 2 + q + nelements * p);

			//float vi = dot(rj, ri, rk, ri); // ji, ki
			float vi = dot_abcd(drji, drki);
			//float vj = dot(rk, rj, ri, rj); // kj, ij
			float vj = dot_abcd(drkj, drij);
			//float vk = dot(ri, rk, rj, rk); // ik, jk
			float vk = dot_abcd(drik, drjk);

			float dcos_angle = zeta_factor * 2.0 * sinf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));
			float dsin_angle = -zeta_factor * 2.0 * cosf(angle) / sqrt(max(1e-10, rij2 * rik2 - vi * vi));

			float angular[2] = { cos_angle, sin_angle };
			float dangular[2] = { dcos_angle, dsin_angle };

			float atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
			float atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
			float atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

			float d_radial_d_j[3];
			float d_radial_d_k[3];
			float d_radial_d_i[3];

			float d_angular_d_j[3];
			float d_angular_d_k[3];
			float d_angular_d_i[3];

			float d_ijdecay[3];
			float d_ikdecay[3];

			float d_atm_ii[3];
			float d_atm_ij[3];
			float d_atm_ik[3];

			float d_atm_ji[3];
			float d_atm_jj[3];
			float d_atm_jk[3];

			float d_atm_ki[3];
			float d_atm_kj[3];
			float d_atm_kk[3];

			float d_atm_extra_i[3];
			float d_atm_extra_j[3];
			float d_atm_extra_k[3];

			for (int x = 0; x < 3; x++) {

				float a = drji[x];
				float b = 0.0;
				float c = drki[x];

				d_radial_d_j[x] = (b - a) * invrij;
				d_radial_d_k[x] = (b - c) * invrik;
				d_radial_d_i[x] = -(d_radial_d_j[x] + d_radial_d_k[x]);

				d_angular_d_j[x] = (c - b) + vi * ((b - a) * invrij2);
				d_angular_d_k[x] = (a - b) + vi * ((b - c) * invrik2);
				d_angular_d_i[x] = -(d_angular_d_j[x] + d_angular_d_k[x]);

				d_ijdecay[x] = -M_PI * (b - a) * sinf(M_PI * rij * invcut) * 0.5 * invrij * invcut;
				d_ikdecay[x] = -M_PI * (b - c) * sinf(M_PI * rik * invcut) * 0.5 * invrik * invcut;

				d_atm_ii[x] = 2 * b - a - c - vi * ((b - a) * invrij2 + (b - c) * invrik2);
				d_atm_ij[x] = c - a - vj * (b - a) * invrij2;
				d_atm_ik[x] = a - c - vk * (b - c) * invrik2;

				d_atm_ji[x] = c - b - vi * (a - b) * invrij2;
				d_atm_jj[x] = 2 * a - b - c - vj * ((a - b) * invrij2 + (a - c) * invrjk2);
				d_atm_jk[x] = b - c - vk * (a - c) * invrjk2;

				d_atm_ki[x] = a - b - vi * (c - b) * invrik2;
				d_atm_kj[x] = b - a - vj * (c - a) * invrjk2;
				d_atm_kk[x] = 2 * c - a - b - vk * ((c - a) * invrjk2 + (c - b) * invrik2);

				d_atm_extra_i[x] = ((a - b) * invrij2 + (c - b) * invrik2) * atm * three_body_decay / three_body_weight;
				d_atm_extra_j[x] = ((b - a) * invrij2 + (c - a) * invrjk2) * atm * three_body_decay / three_body_weight;
				d_atm_extra_k[x] = ((a - c) * invrjk2 + (b - c) * invrik2) * atm * three_body_decay / three_body_weight;
			}

			float deriv_katom[3] = { 0.0, 0.0, 0.0 };

			float rcuts = rcutij * rcutik;

			float atm_cut = atm * rcuts;

			for (int l = 0; l < nRs3; l += 1) {

				float radial = expf(-eta3 * powf(0.5 * (rij + rik) - sRs3[l], 2.0));
				float d_radial = radial * eta3 * (0.5 * (rij + rik) - sRs3[l]);

				float grad_in_iatom_list[2] = { grad_in[molID][iatom][s + l * 2], grad_in[molID][iatom][s + l * 2 + 1] };

				for (int x = 0; x < 3; x++) {

					for (int a = 0; a < 2; a++) {

						float dangular_a = dangular[a];
						float angular_a = angular[a];

						float grad_in_iatom = grad_in_iatom_list[a];

						float deriv_iatomx = dangular_a * d_angular_d_i[x] * radial * atm_cut + angular_a * d_radial * d_radial_d_i[x] * atm_cut
								+ angular_a * radial * (atm_i * d_atm_ii[x] + atm_j * d_atm_ij[x] + atm_k * d_atm_ik[x] + d_atm_extra_i[x]) * three_body_weight
										* rcuts + angular_a * radial * (d_ijdecay[x] * rcutik + rcutij * d_ikdecay[x]) * atm;

						float deriv_jatomx = dangular_a * d_angular_d_j[x] * radial * atm_cut + angular_a * d_radial * d_radial_d_j[x] * atm_cut
								+ angular_a * radial * (atm_i * d_atm_ji[x] + atm_j * d_atm_jj[x] + atm_k * d_atm_jk[x] + d_atm_extra_j[x]) * three_body_weight
										* rcuts - angular_a * radial * d_ijdecay[x] * rcutik * atm;

						float deriv_katomx = dangular_a * d_angular_d_k[x] * radial * atm_cut + angular_a * d_radial * d_radial_d_k[x] * atm_cut
								+ angular_a * radial * (atm_i * d_atm_ki[x] + atm_j * d_atm_kj[x] + atm_k * d_atm_kk[x] + d_atm_extra_k[x]) * three_body_weight
										* rcuts - angular_a * radial * rcutij * d_ikdecay[x] * atm;

						igrad[x] += grad_in_iatom * deriv_iatomx;
						deriv_jatom[x] += grad_in_iatom * deriv_jatomx;
						deriv_katom[x] += grad_in_iatom * deriv_katomx;

					}
				}
			}

			atomicAdd(&sgradx[katom], deriv_katom[0]);
			atomicAdd(&sgrady[katom], deriv_katom[1]);
			atomicAdd(&sgradz[katom], deriv_katom[2]);
		}

		atomicAdd(&sgradx[jatom], deriv_jatom[0]);
		atomicAdd(&sgrady[jatom], deriv_jatom[1]);
		atomicAdd(&sgradz[jatom], deriv_jatom[2]);
	}

	__syncthreads();

	atomicAdd(&grad_out[molID][iatom][0], igrad[0]);
	atomicAdd(&grad_out[molID][iatom][1], igrad[1]);
	atomicAdd(&grad_out[molID][iatom][2], igrad[2]);

	for (int jatom = threadIdx.y * blockDim.x + threadIdx.x; jatom < nneighbours_i; jatom += blockDim.x * blockDim.y) {
		int j = sneighbours[jatom];

		atomicAdd(&grad_out[molID][j][0], sgradx[jatom]);
		atomicAdd(&grad_out[molID][j][1], sgrady[jatom]);
		atomicAdd(&grad_out[molID][j][2], sgradz[jatom]);
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

void getElementTypesCUDA(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor natom_counts, torch::Tensor species, torch::Tensor element_types) {

	int nbatch = coordinates.size(0);
	const int nthreads = 32;

	get_element_types_kernel<<<nbatch, nthreads>>>(coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			natom_counts.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void FCHLCuda_old(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor output) {

	const int nthreadsx = 32;
	const int nthreadsy = 1;

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);
	int nspecies = species.size(0);

//int repsize = nspecies * nRs2 + (nspecies * (nspecies + 1)) * nRs3;

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	dim3 blocks(currBatch);

	dim3 grid(nthreadsx, nthreadsy);

	int shared_mem_size = nRs2 + nRs3 + 4 * max_neighbours + 18; //+ repsize;

	fchl19_representation_cuda<<<blocks, grid, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			Rs2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			Rs3.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			eta2, eta3, two_body_decay, three_body_weight, three_body_decay,rcut,

			output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void FCHLCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell, torch::Tensor inv_cell,
		torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3,
		float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output) {

	const int nthreadsx = 16;
	const int nthreadsy = 8;

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);
	int nspecies = species.size(0);

//int repsize = nspecies * nRs2 + (nspecies * (nspecies + 1)) * nRs3;

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	dim3 blocks(currBatch);

	dim3 grid(nthreadsx, nthreadsy);

	int shared_mem_size = nRs2 + nRs3 + 4 * max_neighbours + 18; //+ repsize;

	fchl19_representation_cuda<<<blocks, grid, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			Rs2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			Rs3.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			eta2, eta3, two_body_decay, three_body_weight, three_body_decay,rcut,

			output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void FCHLDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor grad) {

	const int nthreads = 32;

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);
	int nspecies = species.size(0);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	int shared_mem_size = nRs2 + nRs3 + 5 * max_neighbours;

	fchl19_derivative_cuda<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			Rs2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			Rs3.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			eta2, eta3, two_body_decay, three_body_weight, three_body_decay,rcut,
			grad.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void FCHLBackwardsCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor grad_in, torch::Tensor grad_out) {

	const int nthreadsx = 16;
	const int nthreadsy = 8;

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);
	int nspecies = species.size(0);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	int shared_mem_size = nRs2 + nRs3 + 8 * max_neighbours;

	dim3 blocks(currBatch);

	dim3 grid(nthreadsx, nthreadsy);

	fchl19_backwards_cuda<<<blocks, grid, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			Rs2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			Rs3.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			eta2, eta3, two_body_decay, three_body_weight, three_body_decay,rcut,
			grad_in.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			grad_out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}

void FCHLRepresentationAndDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist,
		torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::Tensor output, torch::Tensor grad) {

	const int nthreads = 32;

	int nRs2 = Rs2.size(0);
	int nRs3 = Rs3.size(0);
	int nspecies = species.size(0);

	const int currBatch = blockAtomIDs.size(0);
	const int max_neighbours = nneighbours.max().item<int>();

	int shared_mem_size = nRs2 + nRs3 + 5 * max_neighbours + 18;

	fchl19_representation_and_derivative_cuda<<<currBatch, nthreads, shared_mem_size * sizeof(float)>>>(
			coordinates.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			charges.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			species.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			element_types.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
			cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			inv_cell.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			blockAtomIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			blockMolIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			neighbourlist.packed_accessor32<int, 3, torch::RestrictPtrTraits>(),
			nneighbours.packed_accessor32<int,2, torch::RestrictPtrTraits>(),
			max_neighbours,
			Rs2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			Rs3.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			eta2, eta3, two_body_decay, three_body_weight, three_body_decay,rcut,

			output.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), grad.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();

}
