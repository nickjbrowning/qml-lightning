#include <torch/extension.h>
#include <iostream>

using namespace at;
using namespace std;

void compute_sorf_matrix(torch::Tensor input, torch::Tensor scaling, torch::Tensor output);

void compute_molecular_featurization(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features);

void compute_molecular_featurization_derivative(torch::Tensor partial_feature_derivatives, double normalisation, torch::Tensor scaling,
		torch::Tensor input_derivatives, torch::Tensor ordering, torch::Tensor feature_derivatives);

void compute_partial_feature_derivatives(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor sin_coeffs);

void hadamard_gpu(torch::Tensor input, torch::Tensor dmatrix, torch::Tensor output, float normalisation, const int ntransforms);
void hadamard_gpu2(torch::Tensor input, torch::Tensor dmatrix, torch::Tensor output, float normalisation, const int ntransforms, const int nthreadsY);

void hadamard_backwards_gpu(torch::Tensor input, torch::Tensor dmatrix, torch::Tensor output, float normalisation, const int ntransforms);

void cos_features_gpu(torch::Tensor coeffs, torch::Tensor b, torch::Tensor batch_indexes, torch::Tensor output);
void cos_derivative_features_gpu(torch::Tensor grads, torch::Tensor coeffs, torch::Tensor b, torch::Tensor batch_indexes, torch::Tensor output);

torch::Tensor CosFeaturesCUDA(torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { nmol, coeffs.size(1) }, options);

	cos_features_gpu(coeffs, b, batch_indexes, output);

	return output;
}

torch::Tensor CosDerivativeFeaturesCUDA(torch::Tensor grads, torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { coeffs.size(0), coeffs.size(1) }, options);

	cos_derivative_features_gpu(grads, coeffs, b, batch_indexes, output);

	return output;
}

torch::Tensor hadamard_transform_gpu(torch::Tensor input, torch::Tensor dmatrix, float normalisation, const int ntransforms) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(dmatrix.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), dmatrix.size(1), input.size(1) }, options);

	hadamard_gpu(input, dmatrix, output, normalisation, ntransforms);

	return output;
}

torch::Tensor hadamard_transform_gpu2(torch::Tensor input, torch::Tensor dmatrix, float normalisation, const int ntransforms, const int nthreadsY) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(dmatrix.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), dmatrix.size(1), input.size(1) }, options);

	hadamard_gpu2(input, dmatrix, output, normalisation, ntransforms, nthreadsY);

	return output;
}

torch::Tensor hadamard_transform_backwards_gpu(torch::Tensor input, torch::Tensor dmatrix, float normalisation, const int ntransforms) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(dmatrix.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), input.size(2) }, options);

	hadamard_backwards_gpu(input, dmatrix, output, normalisation, ntransforms);

	return output;
}

torch::Tensor sorf_matrix_gpu(torch::Tensor input, torch::Tensor scaling) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");

	int natoms = input.size(0);
	int nfeatures = scaling.size(1) * scaling.size(2);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output_sorf_matrix = torch::zeros( { natoms, nfeatures }, options);

	compute_sorf_matrix(input, scaling, output_sorf_matrix);

	return output_sorf_matrix;
}

void compute_hadamard_features(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features) {
	TORCH_CHECK(sorf_matrix.device().type() == torch::kCUDA, "sorf_matrix must be a CUDA tensor");

	compute_molecular_featurization(sorf_matrix, bias, ordering, features);
}

void compute_hadamard_derivative_features(torch::Tensor sorf_matrix, double normalisation, torch::Tensor bias, torch::Tensor scaling,
		torch::Tensor input_derivatives, torch::Tensor ordering, torch::Tensor feature_derivatives) {

//TORCH_CHECK(sorf_matrix.device().type() == torch::kCUDA, "sorf_matrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(torch::kCUDA);

//computes the derivative of the feature only, and not the full chain
	torch::Tensor partial_feature_derivatives = torch::zeros( { sorf_matrix.size(0), sorf_matrix.size(1) }, options);
	compute_partial_feature_derivatives(sorf_matrix, bias, partial_feature_derivatives);

//computes the full chain
	compute_molecular_featurization_derivative(partial_feature_derivatives, normalisation, scaling, input_derivatives, ordering, feature_derivatives);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("sorf_matrix_gpu", &sorf_matrix_gpu, "Computes the SORF matrix components (before featurization).");

	m.def("CosFeaturesCUDA", &CosFeaturesCUDA, "");
	m.def("CosDerivativeFeaturesCUDA", &CosDerivativeFeaturesCUDA, "");
	m.def("hadamard_transform_gpu", &hadamard_transform_gpu, "hadamard transform");
	m.def("hadamard_transform_gpu2", &hadamard_transform_gpu2, "hadamard transform");

	m.def("hadamard_transform_backwards_gpu", &hadamard_transform_backwards_gpu, "hadamard backwards transform");

	m.def("compute_partial_feature_derivatives", &compute_partial_feature_derivatives, "");
	m.def("compute_molecular_featurization_derivative", &compute_molecular_featurization_derivative, "");
	m.def("compute_hadamard_features", &compute_hadamard_features, "Computes the featurisation tensor");
	m.def("compute_hadamard_derivative_features", &compute_hadamard_derivative_features, "Computes the featurisation derivative tensor");
}
