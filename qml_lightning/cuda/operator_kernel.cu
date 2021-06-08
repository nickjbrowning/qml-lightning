#include<torch/torch.h>
#include<cmath>

using namespace std;

__global__ void compute_kmm_scalar_kernel(torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> kmm,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points, double sigma) {

	//extern __shared__ float shared_data[];

	int num_sparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	// K_{ij} = \exp{(-\sum{X_i^M - X_j^M}^2 / 2 \times sigma ^ 2)}

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	double dist = 0.0;
	double invsigma2 = 1.0 / (2 * pow(sigma, 2.0));

	if (idx < num_sparse && idy < num_sparse) {

		for (int k = 0; k < nfeatures; k++) {
			double diff = ((double) sparse_points[idx][k] - (double) sparse_points[idy][k]);
			dist += pow(diff, 2);
		}

		kmm[idx][idy] = exp(-dist * invsigma2);

	}
}

__global__ void compute_knm_scalar_kernel(torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, double sigma) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);
	int nstructs = reps.size(0);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nsparse)
		return;

	double invsigma2 = 1.0 / (2 * pow(sigma, 2.0));

	for (int j = 0; j < nstructs; j++) {

		double dist = 0.0;

		for (int l = 0; l < nfeatures; l++) {
			double diff = ((double) sparse_points[idx][l] - (double) reps[j][l]);
			dist += pow(diff, 2);
		}

		atomicAdd(&knm[indexes[j]][idx], exp(-dist * invsigma2));
	}

}

__global__ void compute_knm_scalar_elemental_kernel(torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rep_atom_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, double sigma) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);
	int natoms = reps.size(0);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nsparse)
		return;

	int sparse_type = sparse_types[idx];

	double invsigma2 = 1.0 / (2 * pow(sigma, 2.0));

	for (int k = 0; k < natoms; k++) {

		double dist = 0.0;

		if (rep_atom_types[k] == sparse_type) {

			for (int l = 0; l < nfeatures; l++) {
				double diff = ((double) sparse_points[idx][l] - (double) reps[k][l]);
				dist += pow(diff, 2);
			}

			double kern = exp(-dist * invsigma2);

			atomicAdd(&knm[indexes[k]][idx], kern);
		}

	}

}

__global__ void compute_knm_scalar_elemental_kernel_poly(torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> knm,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> sparse_points,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> sparse_types,
		const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> reps,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> rep_atom_types,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, double power) {

	int nsparse = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);
	int natoms = reps.size(0);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nsparse)
		return;

	int sparse_type = sparse_types[idx];

	for (int k = 0; k < natoms; k++) {

		double dist = 0.0;

		if (rep_atom_types[k] == sparse_type) {

			for (int l = 0; l < nfeatures; l++) {
				double diff = ((double) sparse_points[idx][l] * (double) reps[k][l]);
				dist += diff;
			}

			double kern = pow(dist, power);

			atomicAdd(&knm[indexes[k]][idx], kern);
		}

	}

}

void compute_knm_elemental_gpu(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_atom_types,
		torch::Tensor indexes, double sigma) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	int nparts = int(ceil(float(npoints) / float(nthreads)));

	dim3 numBlocks(nparts, 1);
	dim3 threadsPerBlock(nthreads, 1);

	compute_knm_scalar_elemental_kernel<<<numBlocks, threadsPerBlock>>>(
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sparse_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			rep_atom_types.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			indexes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma);

	cudaDeviceSynchronize();
}

void compute_kmm_gpu(torch::Tensor kmm, torch::Tensor sparse_points, double sigma) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 32;

	int nparts = int(ceil(float(npoints) / float(nthreads)));

	dim3 numBlocks(nparts, nparts);
	dim3 threadsPerBlock(nthreads, nthreads);

	compute_kmm_scalar_kernel<<<numBlocks, threadsPerBlock>>>(
			kmm.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sigma);

	cudaDeviceSynchronize();
}

void compute_knm_gpu(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor indexes, double sigma) {

	int npoints = sparse_points.size(0);
	int nfeatures = sparse_points.size(1);

	const int nthreads = 128;

	int nparts = int(ceil(float(npoints) / float(nthreads)));

	dim3 numBlocks(nparts, 1);
	dim3 threadsPerBlock(nthreads, 1);

	compute_knm_scalar_kernel<<<numBlocks, threadsPerBlock>>>(
			knm.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			sparse_points.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			reps.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			indexes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			sigma);

	cudaDeviceSynchronize();
}

