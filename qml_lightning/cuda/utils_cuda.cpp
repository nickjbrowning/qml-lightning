#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void mul_in_place_by_const_cuda(torch::Tensor input, float f);
void cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output);
void derivative_cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output);
void outer_product_cuda(torch::Tensor mat, torch::Tensor out);
void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void matmul_and_reduce_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

torch::Tensor matmul_(torch::Tensor A, torch::Tensor B) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor C = torch::zeros( { A.size(0), B.size(1) }, options);

	matmul_cuda(A, B, C);

	return C;

}

void matmul_and_reduce(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
	// A: FP32
	// B: FP32
	// C: FP64

	matmul_and_reduce_cuda(A, B, C);

}

torch::Tensor outer_product(torch::Tensor input) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(1), input.size(1) }, options);

	outer_product_cuda(input, output);

	return output;

}
void MulInPlaceByConstCUDA(torch::Tensor input, float f) {

	mul_in_place_by_const_cuda(input, f);

}

void CosFeaturesCUDA(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output) {
	cos_features_cuda(coeffs, indexes, bias, normalisation, output);
}

void DerivativeCosFeaturesCUDA(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output) {
	derivative_cos_features_cuda(coeffs, indexes, bias, normalisation, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

	m.def("outer_product", &outer_product, "");
	m.def("MulInPlaceByConstCUDA", &MulInPlaceByConstCUDA, "");
	m.def("CosFeaturesCUDA", &CosFeaturesCUDA, "");
	m.def("DerivativeCosFeaturesCUDA", &DerivativeCosFeaturesCUDA, "");
	m.def("matmul", &matmul_, "");
	m.def("matmul_and_reduce", &matmul_and_reduce, "");

}
