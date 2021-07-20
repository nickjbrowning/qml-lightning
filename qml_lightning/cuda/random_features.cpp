#include <torch/extension.h>
#include <iostream>

using namespace at;
using namespace std;

void compute_rff(torch::Tensor input, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features);
void compute_rff_derivatives(torch::Tensor input, torch::Tensor grad, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering,
		torch::Tensor feature_derivative);

void get_rff(torch::Tensor input, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering, torch::Tensor features) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(sampling_matrix.device().type() == torch::kCUDA, "sampling_matrix must be a CUDA tensor");
	TORCH_CHECK(bias.device().type() == torch::kCUDA, "bias must be a CUDA tensor");
	TORCH_CHECK(ordering.device().type() == torch::kCUDA, "ordering must be a CUDA tensor");
	TORCH_CHECK(features.device().type() == torch::kCUDA, "features must be a CUDA tensor");

	compute_rff(input, sampling_matrix, bias, ordering, features);
}

void get_rff_derivatives(torch::Tensor input, torch::Tensor grad, torch::Tensor sampling_matrix, torch::Tensor bias, torch::Tensor ordering,
		torch::Tensor feature_derivatives) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(grad.device().type() == torch::kCUDA, "grad must be a CUDA tensor");
	TORCH_CHECK(sampling_matrix.device().type() == torch::kCUDA, "sampling_matrix must be a CUDA tensor");
	TORCH_CHECK(bias.device().type() == torch::kCUDA, "bias must be a CUDA tensor");
	TORCH_CHECK(ordering.device().type() == torch::kCUDA, "ordering must be a CUDA tensor");
	TORCH_CHECK(feature_derivatives.device().type() == torch::kCUDA, "feature derivatives must be a CUDA tensor");

	compute_rff_derivatives(input, grad, sampling_matrix, bias, ordering, feature_derivatives);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_rff", &get_rff, "Computes kitchen sink features");
	m.def("get_rff_derivatives", &get_rff_derivatives, "Computes kitchen sink features");
}
