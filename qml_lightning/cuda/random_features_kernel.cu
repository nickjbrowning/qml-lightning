#include<torch/torch.h>

using namespace std;

__global__ void random_fourrier_features_kernel(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,
		const torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> sampling_matrix,
		const torch::PackedTensorAccessor32<double, 1, torch::RestrictPtrTraits> bias,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> order,
		torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> output) {

	extern __shared__ float s[];

	int npcas = sampling_matrix.size(0);
	int nfeatures = sampling_matrix.size(1);

	double normalization = sqrt(2.0 / (double) nfeatures);

	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int mol = order[bid];

	for (int i = tid; i < npcas; i += blockDim.x) {
		s[i] = input[bid][i];
	}

	__syncthreads();

	for (int i = tid; i < nfeatures; i += blockDim.x) {

		double sumprod = 0.0;

		for (int j = 0; j < npcas; j++) {
			sumprod += ((double) s[j]) * sampling_matrix[j][i];
		}

		double feature_i = normalization * cos(sumprod + bias[i]);

		atomicAdd(&output[mol][i], feature_i);
	}
}

__global__ void random_fourrier_features_kernel_derivative(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> grad,
		const torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> sampling_matrix,
		const torch::PackedTensorAccessor32<double, 1, torch::RestrictPtrTraits> bias,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> order,
		torch::PackedTensorAccessor32<double, 4, torch::RestrictPtrTraits> grad_output) {

	extern __shared__ float s[];

	int npcas = sampling_matrix.size(0);
	int nfeatures = sampling_matrix.size(1);

	double normalization = sqrt(2.0 / (double) nfeatures);

	int nderiv_atoms = grad.size(1);

	int iatom = int(floor(float(blockIdx.x) / nderiv_atoms));
	int jatom = blockIdx.x % nderiv_atoms;

	int tid = threadIdx.x;

	int mol = order[iatom];

	for (int i = tid; i < npcas; i += blockDim.x) {
		s[i] = input[iatom][i];
	}

	__syncthreads();

	for (int i = tid; i < nfeatures; i += blockDim.x) {

		double sumprod = 0.0;

		for (int j = 0; j < npcas; j++) {
			sumprod += ((double) s[j]) * sampling_matrix[j][i];
		}

		double feature_deriv_i = normalization * sin(sumprod + bias[i]);

		for (int x = 0; x < 3; x++) {

			double sumprod_deriv = 0.0;

			for (int j = 0; j < npcas; j++) {
				sumprod_deriv += ((double) grad[iatom][jatom][x][j] * sampling_matrix[j][i]);

			}

			atomicAdd(&grad_output[mol][jatom][x][i], feature_deriv_i * sumprod_deriv);
		}
	}
}

void compute_rff(torch::Tensor input, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features) {

	int currBatchSize = input.size(0);
	int npcas = input.size(1);
	const int nthreads = 64;

	random_fourrier_features_kernel<<<currBatchSize, nthreads, npcas * sizeof(float)>>>(
			input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			sampling_matrix.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			features.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_rff_derivatives(torch::Tensor input, torch::Tensor grad, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering,
		torch::Tensor feature_derivative) {

	int currBatchSize = grad.size(0) * grad.size(1);

	int npcas = input.size(1);

	const int nthreads = 64;

	random_fourrier_features_kernel_derivative<<<currBatchSize, nthreads, npcas * sizeof(float)>>>(
			input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			sampling_matrix.packed_accessor32<double, 2, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<double, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			feature_derivative.packed_accessor32<double, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}
