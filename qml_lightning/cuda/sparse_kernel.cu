#include<torch/torch.h>
#include<cmath>

using namespace std;

__global__ void kernel_compute_kmm_gaussian(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types, float sigma, bool elemental,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> kmm) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	int idx = blockIdx.x;

	int tid = threadIdx.x;

	extern __shared__ int s[];

	float *sparse_point = (float*) &s;
	float *temp_buffer = &sparse_point[nfeatures];

	double inv_sigma2 = 1.0 / (2.0 * pow(sigma, 2.0));

	if (idx >= nsparse)
		return;

	for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {
		sparse_point[i] = sparse_points[idx][i];
	}

	int sparse_type = sparse_types[idx];

	__syncthreads();

	for (int j = 0; j < nsparse; j++) {

		int atom_type = sparse_types[j];

		if (elemental && sparse_type != atom_type) {
			continue;
		}

		double sum = 0.0;

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff_i = pow((double) sparse_point[i] - (double) sparse_points[j][i], 2.0);

			sum += diff_i;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		for (int size = blockDim.x / 2; size > 0; size /= 2) {

			if (tid < size)
				temp_buffer[tid] += temp_buffer[tid + size];

			__syncthreads();
		}

		if (tid == 0) {

			double dij2 = (double) temp_buffer[0];

			double kernel_val = exp(-(dij2 * inv_sigma2));

			kmm[idx][j] = kernel_val;
		}

		__syncthreads();

	}
}

__global__ void kernel_compute_knm_gaussian(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rep_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> molIDs, float sigma, bool elemental,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm) {

	int nsparse = sparse_points.size(0);
	int npcas = sparse_points.size(1);
	int natoms = reps.size(0);

	extern __shared__ int s[];

	float *srep = (float*) &s;

	int rep_type = rep_types[blockIdx.x];

	float inv_sigma2 = 1.0 / (2.0 * pow(sigma, 2.0));

	for (unsigned int i = threadIdx.x; i < npcas; i += blockDim.x) {
		srep[i] = reps[blockIdx.x][i];
	}

	__syncthreads();

	int molID = molIDs[blockIdx.x];

	for (int i = threadIdx.x; i < nsparse; i += blockDim.x) {

		int sparse_type = sparse_types[i];

		if (!elemental || sparse_type == rep_type) {

			float sum = 0.0;

			for (int j = 0; j < npcas; j++) {

				float diff_i = powf(sparse_points[i][j] - srep[j], 2.0);

				sum += diff_i;

			}

			float kernel_val = expf(-sum * inv_sigma2);

			atomicAdd(&knm[molID][i], (double) kernel_val);

		}
	}
}

void compute_elemental_knm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_types, torch::Tensor molIDs,
		float sigma, torch::Tensor knm) {

	int npoints = reps.size(0);
	int npcas = reps.size(1);

	const int nthreads = 128;

	kernel_compute_knm_gaussian<<<npoints, nthreads, npcas * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			molIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma, true,
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_knm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_types, torch::Tensor molIDs,
		float sigma, torch::Tensor knm) {

	int npoints = reps.size(0);
	int npcas = reps.size(1);

	const int nthreads = 128;

	kernel_compute_knm_gaussian<<<npoints, nthreads, npcas * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			molIDs.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma, false,
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor kmm) {

	int npoints = sparse_points.size(0);
	int npcas = sparse_points.size(1);

	const int nthreads = 128;

	kernel_compute_kmm_gaussian<<<npoints, nthreads, ( npcas + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma, false,
			kmm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_elemental_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor kmm) {

	int npoints = sparse_points.size(0);
	int npcas = sparse_points.size(1);

	const int nthreads = 128;

	kernel_compute_kmm_gaussian<<<npoints, nthreads, ( npcas + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma, true,
			kmm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}
