#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

__global__ void mul_in_place_by_const_kernel(torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> input, float f) {

// torch.Size([8000, 32, 256])

	int atomId = blockIdx.x;

	for (int i = threadIdx.x; i < input.size(1); i += blockDim.x) {
		for (int j = threadIdx.y; j < input.size(2); j += blockDim.y) {
			input[atomId][i][j] *= f;
		}
	}
}

__global__ void cos_features_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coeffs,
		torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> bias,
		float normalisation, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int natoms = coeffs.size(0);
	int nfeatures = coeffs.size(1);

	for (int i = 0; i < blockDim.x; i++) {

		int atom = blockIdx.x * blockDim.x + threadIdx.x;

		if (atom < natoms) {

			int index = indexes[i];

			for (int j = threadIdx.y; j < nfeatures; j += blockDim.y) {

				float val = normalisation * cosf(coeffs[atom][j] + bias[j]);

				atomicAdd(&output[index][j], val);
			}
		}
	}
}

__global__ void deriv_cos_features_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coeffs,
		torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> indexes, torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> bias,
		float normalisation, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {

	extern __shared__ int s[];

	int natoms = coeffs.size(0);
	int nfeatures = coeffs.size(1);

	for (int i = 0; i < blockDim.x; i++) {

		int atom = blockIdx.x * blockDim.x + threadIdx.x;

		if (atom < natoms) {

			int index = indexes[i];

			for (int j = threadIdx.y; j < nfeatures; j += blockDim.y) {

				float val = normalisation * -sinf(coeffs[atom][j] + bias[j]);

				atomicAdd(&output[index][j], val);
			}
		}
	}
}

void cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output) {
//coeffs: natoms, nfeatures
//indexes: natoms
//output: nmols, nfeatures

	const int currBatch = coeffs.size(0);
	const int nthreadsx = 4;
	const int nthreadsy = 32;

	dim3 blocks(currBatch / nthreadsx);

	dim3 grid(nthreadsx, nthreadsy);

	cos_features_kernel<<<blocks, grid>>>(
			coeffs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			indexes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			normalisation,
			output.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
	);

	cudaDeviceSynchronize();
}

void derivative_cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output) {

	const int currBatch = coeffs.size(0);
	const int nthreadsx = 4;
	const int nthreadsy = 32;

	dim3 blocks(currBatch / nthreadsx);

	dim3 grid(nthreadsx, nthreadsy);

	deriv_cos_features_kernel<<<blocks, grid>>>(
			coeffs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			indexes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			normalisation,
			output.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
	);

	cudaDeviceSynchronize();
}

void mul_in_place_by_const_cuda(torch::Tensor input, float f) {

	const int currBatch = input.size(0);
	const int nthreadsx = 16;
	const int nthreadsy = 32;

	dim3 blocks(currBatch);

	dim3 grid(nthreadsx, nthreadsy);

	mul_in_place_by_const_kernel<<<blocks, grid>>>(
			input.packed_accessor32<float, 3, torch::RestrictPtrTraits>(), f
	);

	cudaDeviceSynchronize();

}

