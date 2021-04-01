#include <torch/extension.h>

using namespace at;

void compute_sorf_matrix_gpu_float(torch::Tensor input, torch::Tensor scaling, torch::Tensor output);

void compute_molecular_featurization_gpu_float(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features);

void compute_molecular_featurization_derivative_gpu_float(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor scaling, torch::Tensor input_derivatives,
		torch::Tensor ordering, torch::Tensor feature_derivatives);

torch::Tensor sorf_matrix_gpu(torch::Tensor input, torch::Tensor scaling, int nfeatures) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");

	int natoms = input.size(0);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output_sorf_matrix = torch::zeros( { natoms, nfeatures }, options);

	compute_sorf_matrix_gpu_float(input, scaling, output_sorf_matrix);

	return output_sorf_matrix;
}

torch::Tensor molecular_featurisation_gpu(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor ordering, int nbatch) {
	TORCH_CHECK(sorf_matrix.device().type() == torch::kCUDA, "sorf_matrix must be a CUDA tensor");

	int natoms = sorf_matrix.size(0);
	int nfeatures = sorf_matrix.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor features = torch::zeros( { nbatch, nfeatures }, options);

	compute_molecular_featurization_gpu_float(sorf_matrix, bias, ordering, features);

	return features;
}

torch::Tensor molecular_featurisation_derivative_gpu(torch::Tensor sorf_matrix, torch::Tensor bias, torch::Tensor scaling, torch::Tensor input_derivatives,
		torch::Tensor ordering, int nbatch) {

	TORCH_CHECK(sorf_matrix.device().type() == torch::kCUDA, "sorf_matrix must be a CUDA tensor");

	int natoms_deriv = input_derivatives.size(1);
	int nfeatures = sorf_matrix.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor feature_derivatives = torch::zeros( { nbatch, natoms_deriv, 3, nfeatures }, options);

	compute_molecular_featurization_derivative_gpu_float(sorf_matrix, bias, scaling, input_derivatives, ordering, feature_derivatives);

	return feature_derivatives;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("sorf_matrix_gpu", &sorf_matrix_gpu, "Computes the SORF matrix components (before featurization).");
	m.def("molecular_featurisation_gpu", &molecular_featurisation_gpu, "Computes the featurisation tensor");
	m.def("molecular_featurisation_derivative_gpu", &molecular_featurisation_derivative_gpu, "Computes the featurisation derivative tensor");
}
