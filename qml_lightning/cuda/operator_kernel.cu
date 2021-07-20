#include<torch/torch.h>
#include<cmath>

using namespace std;

__global__ void kernel_compute_kmm_gaussian_shared(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points, float sigma,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> kmm) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	int idx = blockIdx.x;

	int tid = threadIdx.x;

	extern __shared__ int s[];

	float *sparse_point = (float*) &s;
	float *temp_buffer = &sparse_point[nfeatures];

	float inv_sigma2 = 1.0 / (2.0 * pow(sigma, 2.0));

	if (idx >= nsparse)
		return;

	for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {
		sparse_point[i] = sparse_points[idx][i];
	}

	__syncthreads();

	for (int j = 0; j < nsparse; j++) {

		double sum = 0.0;

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff_i = pow((double) sparse_point[i] - (double) sparse_points[j][i], 2.0);

			sum += diff_i;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		//now reduce dot_products_components in shared memory
		//0: size = 64 / 2 -> 32
		//1: size = 32/ 2 -> 16
		//2: 8
		//3: 4

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

__global__ void kernel_compute_knm_gaussian_shared(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, float sigma,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);
	int natoms = reps.size(0);

	int idx = blockIdx.x;

	int tid = threadIdx.x;

	extern __shared__ int s[];

	float *sparse_point = (float*) &s;
	float *temp_buffer = &sparse_point[nfeatures];

	float inv_sigma2 = 1.0 / (2.0 * pow(sigma, 2.0));

	if (idx >= nsparse)
		return;

	for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {
		sparse_point[i] = sparse_points[idx][i];
	}

	__syncthreads();

	for (int j = 0; j < natoms; j++) {

		int batchID = indexes[j];

		double sum = 0.0;

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff_i = pow((double) sparse_point[i] - (double) reps[j][i], 2.0);

			sum += diff_i;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		//now reduce dot_products_components in shared memory
		//0: size = 64 / 2 -> 32
		//1: size = 32/ 2 -> 16
		//2: 8
		//3: 4

		for (int size = blockDim.x / 2; size > 0; size /= 2) {

			if (tid < size)
				temp_buffer[tid] += temp_buffer[tid + size];

			__syncthreads();
		}

		if (tid == 0) {
			//double dij = sqrt((double) temp_buffer[0]);
			double dij2 = (double) temp_buffer[0];
			//printf("dij: %f %d\n", dij, blockIdx.x);

			double kernel_val = exp(-(dij2 * inv_sigma2));

			atomicAdd(&knm[batchID][idx], kernel_val);
		}

		__syncthreads();

	}
}

__global__ void kernel_compute_knm_gaussian_shared_derivative(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rep_grad,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ordering, float sigma,
		torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> knm_derivative) {

	extern __shared__ int s[];

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	int nderiv_atoms = rep_grad.size(1);

	int iatom = int(floor(float(blockIdx.x) / nderiv_atoms));
	int jatom = blockIdx.x % nderiv_atoms;

	int batchID = ordering[iatom];

	int tid = threadIdx.x;

	float *rep_iatom = (float*) &s;
	float *diffs = (float*) &rep_iatom[nfeatures];
	float *temp_buffer = &diffs[nfeatures];

	float sigma2 = pow(sigma, 2.0);
	float inv_sigma2 = 1.0 / (2 * sigma2);

	for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {
		rep_iatom[i] = reps[iatom][i];
	}

	__syncthreads();

	for (int k = 0; k < nsparse; k++) {

		double sum = 0.0;

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff = (double) sparse_points[k][i] - (double) rep_iatom[i];

			double pow_diff = pow(diff, 2);

			diffs[i] = diff;

			sum += pow_diff;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		//now reduce diffs in shared memory
		//0: size = 64 / 2 -> 32
		//1: size = 32/ 2 -> 16
		//2: 8
		//3: 4
		for (int size = blockDim.x / 2; size > 0; size >>= 1) {

			if (tid < size)
				temp_buffer[tid] += temp_buffer[tid + size];

			__syncthreads();
		}

		double dij2 = (double) temp_buffer[0];
		double dij = sqrt(dij2);

		double dkdu = -2.0 * inv_sigma2 * exp(-(dij2 * inv_sigma2));

		for (int x = 0; x < 3; x++) {

			double tmp_sum = 0.0;

			for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

				double dkdr = dkdu * diffs[i] * -rep_grad[iatom][jatom][x][i];

				tmp_sum += dkdr;

			}

			temp_buffer[tid] = tmp_sum;

			__syncthreads();

			for (int size = blockDim.x / 2; size > 0; size >>= 1) {

				if (tid < size)
					temp_buffer[tid] += temp_buffer[tid + size];

				__syncthreads();
			}

			if (tid == 0) {
				atomicAdd(&knm_derivative[batchID][jatom][x][k], temp_buffer[0]);
			}
		}
	}
}

__global__ void kernel_compute_elemental_knm_gaussian_shared(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rep_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, float sigma,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);
	int natoms = reps.size(0);

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

	for (int j = 0; j < natoms; j++) {

		int atom_type = rep_types[j];

		if (sparse_type != atom_type) {
			continue;
		}

		int batchID = indexes[j];

		double sum = 0.0;

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff_i = pow((double) sparse_point[i] - (double) reps[j][i], 2.0);

			sum += diff_i;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		//now reduce dot_products_components in shared memory
		//0: size = 64 / 2 -> 32
		//1: size = 32/ 2 -> 16
		//2: 8
		//3: 4

		for (int size = blockDim.x / 2; size > 0; size /= 2) {

			if (tid < size)
				temp_buffer[tid] += temp_buffer[tid + size];

			__syncthreads();
		}

		if (tid == 0) {
			//double dij = sqrt((double) temp_buffer[0]);
			double dij2 = (double) temp_buffer[0];
			//printf("dij: %f %d\n", dij, blockIdx.x);

			double kernel_val = exp(-(dij2 * inv_sigma2));

			atomicAdd(&knm[batchID][idx], kernel_val);
		}

		__syncthreads();

	}
}

__global__ void kernel_compute_elemental_kmm_gaussian_shared(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types, float sigma,
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

		if (sparse_type != atom_type) {
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

__global__ void kernel_compute_elemental_knm_gaussian_shared_derivative(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rep_grad,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rep_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ordering, float sigma,
		torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> knm_derivative) {

	extern __shared__ int s[];

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	int nderiv_atoms = rep_grad.size(1);

	int iatom = int(floor(float(blockIdx.x) / nderiv_atoms));
	int jatom = blockIdx.x % nderiv_atoms;

	int batchID = ordering[iatom];

	int tid = threadIdx.x;

	float *rep_iatom = (float*) &s;
	float *diffs = (float*) &rep_iatom[nfeatures];
	float *temp_buffer = &diffs[nfeatures];

	double sigma2 = pow(sigma, 2.0);
	double inv_sigma2 = 1.0 / (2.0 * sigma2);

	for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {
		rep_iatom[i] = reps[iatom][i];
	}

	__syncthreads();

	int rep_type = rep_types[iatom];

	for (int k = 0; k < nsparse; k++) {

		double sum = 0.0;

		int atom_type = sparse_types[k];

		if (atom_type != rep_type) {
			continue;
		}

		for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

			double diff = (double) sparse_points[k][i] - (double) rep_iatom[i];

			double pow_diff = pow(diff, 2);

			diffs[i] = diff;

			sum += pow_diff;

		}

		temp_buffer[tid] = sum;

		__syncthreads();

		//now reduce diffs in shared memory
		//0: size = 64 / 2 -> 32
		//1: size = 32/ 2 -> 16
		//2: 8
		//3: 4
		for (int size = blockDim.x / 2; size > 0; size >>= 1) {

			if (tid < size)
				temp_buffer[tid] += temp_buffer[tid + size];

			__syncthreads();
		}

		double dij2 = (double) temp_buffer[0];

		double dkdu = -2.0 * inv_sigma2 * exp(-(dij2 * inv_sigma2));

		for (int x = 0; x < 3; x++) {

			double tmp_sum = 0.0;

			for (unsigned int i = tid; i < nfeatures; i += blockDim.x) {

				double val = diffs[i] * (double) -rep_grad[iatom][jatom][x][i];

				tmp_sum += val;

			}

			temp_buffer[tid] = tmp_sum;

			__syncthreads();

			for (int size = blockDim.x / 2; size > 0; size >>= 1) {

				if (tid < size)
					temp_buffer[tid] += temp_buffer[tid + size];

				__syncthreads();
			}

			if (tid == 0) {
				atomicAdd(&knm_derivative[batchID][jatom][x][k], dkdu * (double) temp_buffer[0]);
			}
		}
	}
}

void compute_kmm_gaussian(torch::Tensor sparse_points, float sigma, torch::Tensor kmm) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	dim3 numBlocks(npoints, 1);
	dim3 threadsPerBlock(nthreads, 1);

	kernel_compute_kmm_gaussian_shared<<<numBlocks, threadsPerBlock, ( nfeatures + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sigma,
			kmm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_knm_gaussian(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor ordering, float sigma, torch::Tensor knm) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	dim3 numBlocks(npoints, 1);
	dim3 threadsPerBlock(nthreads, 1);

	kernel_compute_knm_gaussian_shared<<<numBlocks, threadsPerBlock, ( nfeatures + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma,
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor rep_grad, torch::Tensor ordering, float sigma,
		torch::Tensor knm_derivative) {

	int currBatchSize = rep_grad.size(0) * rep_grad.size(1);

	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	dim3 numBlocks(currBatchSize, 1);
	dim3 threadsPerBlock(nthreads, 1);

	kernel_compute_knm_gaussian_shared_derivative<<<numBlocks, threadsPerBlock, ( (2 * nfeatures) + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma,
			knm_derivative.packed_accessor32<double,4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_elemental_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor kmm) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	kernel_compute_elemental_kmm_gaussian_shared<<<npoints, nthreads, ( nfeatures + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma,
			kmm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_elemental_knm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_types,
		torch::Tensor ordering, float sigma, torch::Tensor knm) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	kernel_compute_elemental_knm_gaussian_shared<<<npoints, nthreads, ( nfeatures + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma,
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_elemental_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_grad,
		torch::Tensor rep_types, torch::Tensor ordering, float sigma, torch::Tensor knm_derivative) {

	int currBatchSize = rep_grad.size(0) * rep_grad.size(1);

	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	kernel_compute_elemental_knm_gaussian_shared_derivative<<<currBatchSize, nthreads, ( (2 * nfeatures) + nthreads) * sizeof(float)>>>(
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			rep_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma,
			knm_derivative.packed_accessor32<double,4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

