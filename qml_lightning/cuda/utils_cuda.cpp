#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void mul_in_place_by_const_cuda(torch::Tensor input, float f);
void cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output);
void derivative_cos_features_cuda(torch::Tensor coeffs, torch::Tensor indexes, torch::Tensor bias, float normalisation, torch::Tensor output);

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
	m.def("MulInPlaceByConstCUDA", &MulInPlaceByConstCUDA, "");
	m.def("CosFeaturesCUDA", &CosFeaturesCUDA, "");
	m.def("DerivativeCosFeaturesCUDA", &DerivativeCosFeaturesCUDA, "");

}
