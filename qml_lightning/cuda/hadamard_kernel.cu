#include<torch/torch.h>

using namespace std;

__global__ void sorf_matrix_kernel_float(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> input,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> D, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output,
		int nstacks, int log2N) {

	/**
	 * Computes the structured orthogonal matrix W from [HD]_n, where D is a rademacher-distributed diagonal matrix
	 * and H is the Hadamard matrix. n corresponds to the number of [HD] operations to perform.
	 *
	 * input is the [natoms, repsize] representation matrix. This should be subselected from the full representation based on element types
	 * such that each element type is transformed in the same way via element-specific D's.
	 *
	 * output is the [natoms, nfeatures] dot product matrix [Wx], where each column of W has been stacked N/d times.
	 *
	 * D is the [n, nstacks, d] rademacher tensor.
	 *
	 * **/

	const int N = 1 << log2N;

	extern __shared__ float s[];

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("check: %d %f", N, powf(2.0, float(log2N) / 2));

	int mdiag = D.size(0); // number of [HD] blocks to compute
	//loop over N/d hadamard transforms to create length-N feature vector
	for (int stack = 0; stack < nstacks; stack++) {

		for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
			s[pos] = input[blockIdx.x][pos];
		}

		//loop over n [HD] blocks
		for (int m = 0; m < mdiag; m++) {

			for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
				s[pos] = D[m][stack][pos] * s[pos];
			}

			/**Hadamard transform taken from Nvidia Cuda Examples**/

			int stride = 1;

			//Do single radix-2 stage for odd power of two
			if (log2N & 1) {

				__syncthreads();

				for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
					int i0 = pos << 1;
					int i1 = i0 + 1;

					float D0 = s[i0];
					float D1 = s[i1];
					s[i0] = D0 + D1;
					s[i1] = D0 - D1;
				}
				stride <<= 1;
			}

			//Main radix-4 stages
			const int pos = threadIdx.x;

			for (; stride <= N >> 2; stride <<= 2) {
				int lo = pos & (stride - 1);
				int i0 = ((pos - lo) << 2) + lo;
				int i1 = i0 + stride;
				int i2 = i1 + stride;
				int i3 = i2 + stride;

				__syncthreads();

				float D0 = s[i0];
				float D1 = s[i1];
				float D2 = s[i2];
				float D3 = s[i3];

				float T;
				T = D0;
				D0 = D0 + D2;
				D2 = T - D2;
				T = D1;
				D1 = D1 + D3;
				D3 = T - D3;
				T = D0;
				s[i0] = D0 + D1;
				s[i1] = T - D1;
				T = D2;
				s[i2] = D2 + D3;
				s[i3] = T - D3;
			}

			__syncthreads();

			/**Finished Hadamard transform for subblock N/d.*/

			//normalize hadamard transform
			for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
				s[pos] = (1.0 / powf(2.0, float(log2N) / 2)) * s[pos];
			}
		}

		__syncthreads();

		//save [HD]n stack to global memory
		for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
			output[blockIdx.x][stack * N + pos] = s[pos];
		}
	}
}

__global__ void compute_featurisation_float(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coefficients,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> bias,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ordering,
		torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> features) {

	//coefficients: natoms, nfeatures
	//features nbatch, nfeatures
	//ordering: contains the indexes of which nbatch to add atom j to.

	int nfeatures = coefficients.size(1);
	int natoms = coefficients.size(0);

	int iatom = blockIdx.x;

	int batchID = ordering[iatom];

	for (int N = threadIdx.x; N < nfeatures; N += blockDim.x) {
		atomicAdd(&features[batchID][N], cos(coefficients[iatom][N] + bias[N]) * sqrt(2.0 / float(nfeatures)));
	}
}

__global__ void compute_featurisation_derivative_float(const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coefficients,
		const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> bias,
		const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> ordering,
		const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> input_derivative,
		const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> D, int nstacks, int log2N,
		torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> feature_derivatives) {

	const int N = 1 << log2N;

	int nfeatures = coefficients.size(1);

	extern __shared__ float u[];

	int mdiag = D.size(0); // number of [HD] blocks to compute
//loop over N/d hadamard transforms to create length-N feature vector

	int nderiv_atoms = input_derivative.size(1);

	int iatom = int(floor(float(blockIdx.x) / nderiv_atoms));
	int jatom = blockIdx.x % nderiv_atoms;

	int batchID = ordering[iatom];

	//printf("thread %d block %d iatom %d jatom %d batchID %d nstacks %d\n", threadIdx.x, blockIdx.x, iatom, jatom, batchID, nstacks);

	for (int x = 0; x < 3; x++) {

		for (int stack = 0; stack < nstacks; stack++) {

			for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
				u[pos] = input_derivative[iatom][jatom][x][pos];
			}

			__syncthreads();

			//loop over n [HD] blocks
			for (int m = 0; m < mdiag; m++) {

				for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
					u[pos] = D[m][stack][pos] * u[pos];
				}

				__syncthreads();

				/**Hadamard transform taken from Nvidia Cuda Examples**/

				int stride = 1;

				//Do single radix-2 stage for odd power of two
				if (log2N & 1) {

					__syncthreads();

					for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
						int i0 = pos << 1;
						int i1 = i0 + 1;

						float D0 = u[i0];
						float D1 = u[i1];
						u[i0] = D0 + D1;
						u[i1] = D0 - D1;
					}
					stride <<= 1;
				}

				//Main radix-4 stages
				const int pos = threadIdx.x;

				for (; stride <= N >> 2; stride <<= 2) {
					int lo = pos & (stride - 1);
					int i0 = ((pos - lo) << 2) + lo;
					int i1 = i0 + stride;
					int i2 = i1 + stride;
					int i3 = i2 + stride;

					__syncthreads();

					float D0 = u[i0];
					float D1 = u[i1];
					float D2 = u[i2];
					float D3 = u[i3];

					float T;
					T = D0;
					D0 = D0 + D2;
					D2 = T - D2;
					T = D1;
					D1 = D1 + D3;
					D3 = T - D3;
					T = D0;

					u[i0] = D0 + D1;
					u[i1] = T - D1;
					T = D2;
					u[i2] = D2 + D3;
					u[i3] = T - D3;
				}

				__syncthreads();

				/**Finished Hadamard transform for subblock N/d.*/

				//normalize hadamard transform
				for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
					u[pos] = (1.0 / powf(2.0, float(log2N) / 2.0)) * u[pos];
				}
			}

			__syncthreads();

			//save d/dr cos([(HD)n] x + b)  stack to global memory
			for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
				float val = -sin(coefficients[iatom][stack * N + pos] + bias[stack * N + pos]) * u[pos] * sqrt(2.0 / float(nfeatures));

				atomicAdd(&feature_derivatives[batchID][jatom][x][stack * N + pos], val);
			}
		}
	}
}

void compute_sorf_matrix_gpu_float(torch::Tensor representations, torch::Tensor scaling, torch::Tensor sorf_matrix) {

	int n = representations.size(1);
	int log2N = int(log2(n));

	int curBatchSize = representations.size(0);

	int nfeatures = sorf_matrix.size(1);

	int log2f = int(log2(nfeatures));

	TORCH_CHECK(n == 1 << log2N, "representation size must be power of 2.");
	TORCH_CHECK(nfeatures == 1 << log2f, "features size must be power of 2.");

	int nstacks = int(float(nfeatures) / n);

	sorf_matrix_kernel_float<<<curBatchSize, (n+3)/4, n * sizeof(float)>>>(representations.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			scaling.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			sorf_matrix.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), nstacks, log2N);

	cudaDeviceSynchronize();
}

void compute_molecular_featurization_gpu_float(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features) {

	int currBatchSize = sorf_matrix.size(0);
	const int nthreads = 32;

	compute_featurisation_float<<<currBatchSize, nthreads>>>(
			sorf_matrix.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			features.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}

void compute_molecular_featurization_derivative_gpu_float(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor scaling, torch::Tensor input_derivatives,
		torch::Tensor ordering, torch::Tensor feature_derivatives) {

	int n = input_derivatives.size(3);
	int log2N = int(log2(n));

	int currBatchSize = input_derivatives.size(0) * input_derivatives.size(1);
//int currBatchSize = input_derivatives.size(0);
	int nfeatures = sorf_matrix.size(1);

	int log2f = int(log2(nfeatures));

	TORCH_CHECK(n == 1 << log2N, "input_derivatives size must be power of 2.");
	TORCH_CHECK(nfeatures == 1 << log2f, "features size must be power of 2.");

	int nstacks = int(float(nfeatures) / n);

	compute_featurisation_derivative_float<<<currBatchSize, (n+3)/4, n * sizeof(float)>>>(
			sorf_matrix.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
			bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
			ordering.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
			input_derivatives.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
			scaling.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
			nstacks, log2N,
			feature_derivatives.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

	cudaDeviceSynchronize();
}
