#include <math.h>
#include<torch/torch.h>
#include <iostream>

using namespace std;

const int BLOCK_SIZE = 16;

__global__ void MatMul_shared_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> A,
		torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> B, torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> C) {

	float sum = 0.0;
	float c = 0.0;

	int ARows = A.size(0);
	int ACols = A.size(1);
	int BRows = B.size(0);
	int BCols = B.size(1);
	int CRows = C.size(0);
	int CCols = C.size(1);

	int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	__shared__
	float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__
	float Bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; k++) {

		if (k * BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
			As[threadIdx.y][threadIdx.x] = A[Row][k * BLOCK_SIZE + threadIdx.x];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		if (k * BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
			Bs[threadIdx.y][threadIdx.x] = B[k * BLOCK_SIZE + threadIdx.y][Col];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < BLOCK_SIZE; ++n) {
			float y = fma(As[threadIdx.y][n], Bs[n][threadIdx.x], -c);
			float t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}

		__syncthreads();
	}

	if (Row < CRows && Col < CCols) {
		//C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = sum;
		C[blockIdx.y * blockDim.y + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x] = sum;
	}
}

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.size(1) / BLOCK_SIZE, A.size(0) / BLOCK_SIZE);

MatMul_shared_kernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
		B.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), C.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

}

__global__ void matmul_and_reduce_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> A,
	torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> B, torch::PackedTensorAccessor32<double, 2, torch::RestrictPtrTraits> C) {

float sum = 0.0;
float c = 0.0;

int ARows = A.size(0);
int ACols = A.size(1);
int BRows = B.size(0);
int BCols = B.size(1);
int CRows = C.size(0);
int CCols = C.size(1);

int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

__shared__
float As[BLOCK_SIZE][BLOCK_SIZE];
__shared__
float Bs[BLOCK_SIZE][BLOCK_SIZE];

for (int k = 0; k < (BLOCK_SIZE + ACols - 1) / BLOCK_SIZE; k++) {

	if (k * BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
		As[threadIdx.y][threadIdx.x] = A[Row][k * BLOCK_SIZE + threadIdx.x];
	else
		As[threadIdx.y][threadIdx.x] = 0.0;

	if (k * BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
		Bs[threadIdx.y][threadIdx.x] = B[k * BLOCK_SIZE + threadIdx.y][Col];
	else
		Bs[threadIdx.y][threadIdx.x] = 0.0;

	__syncthreads();

	for (int n = 0; n < BLOCK_SIZE; ++n) {
		float y = fma(As[threadIdx.y][n], Bs[n][threadIdx.x], -c);
		float t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	__syncthreads();
}

if (Row < CRows && Col < CCols) {
	//C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = sum;
	C[blockIdx.y * blockDim.y + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x] += (double) sum;
}
}

void matmul_and_reduce_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid(B.size(1) / BLOCK_SIZE, A.size(0) / BLOCK_SIZE);

matmul_and_reduce_kernel<<<dimGrid, dimBlock, BLOCK_SIZE * BLOCK_SIZE * sizeof(float)>>>(A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
	B.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), C.packed_accessor32<double, 2, torch::RestrictPtrTraits>());

}

__global__ void outerprod_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> mat1,
torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {

const int i = blockDim.x * blockIdx.x + threadIdx.x;
const int j = blockDim.y * blockIdx.y + threadIdx.y;

int m = mat1.size(1);
int kk = mat1.size(0);
int n = mat1.size(1);

if (i > m - 1 || j > n - 1)
return;

float sum = 0.0;
float c = 0.0;

for (int k = 0; k < kk; ++k) {
//float prod = mat1[k][i] * mat1[k][j];
//float y = prod - c;

float y = fma(mat1[k][i], mat1[k][j], -c);
float t = sum + y;
c = (t - sum) - y;
sum = t;

//sum += ((double) mat1[k][i]) * ((double) mat1[k][j]);
}

output[i][j] = sum;

}

__global__ void outerprod_kernel2(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> mat1,
torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output) {

const int i = blockDim.x * blockIdx.x + threadIdx.x;
const int j = blockDim.y * blockIdx.y + threadIdx.y;

int m = mat1.size(1);
int kk = mat1.size(0);
int n = mat1.size(1);

if (i > m - 1 || j > n - 1)
return;

float sum = 0.0;
float c = 0.0;

for (int k = 0; k < kk; ++k) {
float prod = mat1[k][i] * mat1[k][j];
//float y = prod - c;

float t = sum + prod;

if (abs(sum) >= abs(prod)) {
	c += (sum - t) + prod;
} else {
	c += (prod - t) + sum;
}

sum = t;

}

output[i][j] = sum + c;

}

inline __host__ __device__ int iDivUp(const int& a, const int& b)
{
int result = a % b != 0 ? (a < b ? 1 : a / b + 1) : a / b;
return result;
}

void outer_product_cuda(torch::Tensor mat, torch::Tensor out) {

const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

const int n = mat.size(1);
const int m = mat.size(1);

const dim3 gridDim(iDivUp(n, blockDim.x), iDivUp(m, blockDim.y));

outerprod_kernel<<<gridDim,blockDim>>>
(mat.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

cudaDeviceSynchronize();
}

__global__
void mul_in_place_by_const_kernel(torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> input, float f) {

// torch.Size([8000, 32, 256])

int atomId = blockIdx.x;

for (int i = threadIdx.x; i < input.size(1); i += blockDim.x) {
for (int j = threadIdx.y; j < input.size(2); j += blockDim.y) {
	input[atomId][i][j] *= f;
}
}
}

__global__
void cos_features_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coeffs,
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

__global__
void deriv_cos_features_kernel(torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> coeffs,
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

