#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void compute_knm_gaussian(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor ordering, float sigma, torch::Tensor knm);
void compute_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor rep_grad, torch::Tensor ordering, float sigma,
		torch::Tensor knm_derivatives);

void compute_elemental_knm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_types,
		torch::Tensor ordering, float sigma, torch::Tensor knm);

void compute_elemental_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_grad,
		torch::Tensor rep_types, torch::Tensor ordering, float sigma, torch::Tensor knm_derivative);

void compute_elemental_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor knm);

void compute_kmm_gaussian(torch::Tensor sparse_points, float sigma, torch::Tensor knm);

void compute_elemental_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor kmm);

void get_kmm_gaussian(torch::Tensor sparse_points, float sigma, torch::Tensor kmm) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");

	compute_kmm_gaussian(sparse_points, sigma, kmm);

}

void get_knm_gaussian(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor indexes, float sigma, torch::Tensor knm) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");

	compute_knm_gaussian(sparse_points, reps, indexes, sigma, knm);

}

void get_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor grads, torch::Tensor indexes, float sigma,
		torch::Tensor knm_derivative) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");

	compute_knm_gaussian_derivative(sparse_points, reps, grads, indexes, sigma, knm_derivative);

}

void get_elemental_kmm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, float sigma, torch::Tensor kmm) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");

	compute_elemental_kmm_gaussian(sparse_points, sparse_types, sigma, kmm);

}

void get_elemental_knm_gaussian(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_types, torch::Tensor indexes,
		float sigma, torch::Tensor knm) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");

	compute_elemental_knm_gaussian(sparse_points, sparse_types, reps, rep_types, indexes, sigma, knm);

}

void get_elemental_knm_gaussian_derivative(torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor grads,
		torch::Tensor rep_types, torch::Tensor indexes, float sigma, torch::Tensor knm_derivative) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");

	compute_elemental_knm_gaussian_derivative(sparse_points, sparse_types, reps, grads, rep_types, indexes, sigma, knm_derivative);

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_kmm_gaussian", &get_kmm_gaussian, "K_{MM} gaussian matrix");
	m.def("get_knm_gaussian", &get_knm_gaussian, "K_{NM} gaussian matrix");
	m.def("get_knm_gaussian_derivative", &get_knm_gaussian_derivative, "K_{NM} gausian derivative tensor");

	m.def("get_elemental_kmm_gaussian", &get_elemental_kmm_gaussian, "K_{MM} elemental gaussian matrix");
	m.def("get_elemental_knm_gaussian", &get_elemental_knm_gaussian, "K_{NM} elemental gaussian matrix");
	m.def("get_elemental_knm_gaussian_derivative", &get_elemental_knm_gaussian_derivative, "K_{NM} elemental gaussian derivative tensor");

}
