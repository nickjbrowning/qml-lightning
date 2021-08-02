#include <torch/extension.h>
#include <iostream>

using namespace at;
using namespace std;

void compute_sorf_matrix(torch::Tensor input, torch::Tensor scaling, torch::Tensor output);

void compute_molecular_featurization(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features);

void compute_molecular_featurization_derivative(torch::Tensor partial_feature_derivatives, double normalisation, torch::Tensor scaling,
		torch::Tensor input_derivatives, torch::Tensor ordering, torch::Tensor feature_derivatives);

void compute_partial_feature_derivatives(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor sin_coeffs);

void hadamard_gpu(torch::Tensor input, torch::Tensor output);

torch::Tensor hadamard_transform_gpu(torch::Tensor input) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), input.size(1), input.size(2) }, options);

	hadamard_gpu(input, output);

	return output;
}

torch::Tensor sorf_matrix_gpu(torch::Tensor input, torch::Tensor scaling, int nfeatures) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");

	int natoms = input.size(0);

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
	m.def("hadamard_transform_gpu", &hadamard_transform_gpu, "hadamard transform");
	m.def("compute_partial_feature_derivatives", &compute_partial_feature_derivatives, "");
	m.def("compute_molecular_featurization_derivative", &compute_molecular_featurization_derivative, "");
	m.def("compute_hadamard_features", &compute_hadamard_features, "Computes the featurisation tensor");
	m.def("compute_hadamard_derivative_features", &compute_hadamard_derivative_features, "Computes the featurisation derivative tensor");
}
