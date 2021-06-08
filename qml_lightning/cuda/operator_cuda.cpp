#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void compute_knm_gpu(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor indexes, double sigma);
void compute_kmm_gpu(torch::Tensor kmm, torch::Tensor sparse_points, double sigma);

void compute_knm_elemental_gpu(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps, torch::Tensor rep_atom_types,
		torch::Tensor indexes, double sigma);

torch::Tensor compute_kmm(torch::Tensor sparse_points, double sigma) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");

	int nsparse = sparse_points.size(0);

	auto options = torch::TensorOptions().dtype(torch::kFloat64).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor kmm = torch::zeros( { nsparse, nsparse }, options);

	compute_kmm_gpu(kmm, sparse_points, sigma);

	return kmm;
}

void compute_knm(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor reps, torch::Tensor indexes, double sigma) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");

	int nsparse = sparse_points.size(0);
	int nsystems = reps.size(0);

	compute_knm_gpu(knm, sparse_points, reps, indexes, sigma);

}

void compute_knm_elemental_contribution(torch::Tensor knm, torch::Tensor sparse_points, torch::Tensor sparse_types, torch::Tensor reps,
		torch::Tensor rep_atom_types, torch::Tensor indexes, double sigma) {

	TORCH_CHECK(sparse_points.device().type() == torch::kCUDA, "sparse_points must be a CUDA tensor");
	TORCH_CHECK(sparse_types.device().type() == torch::kCUDA, "sparse_types must be a CUDA tensor");
	TORCH_CHECK(reps.device().type() == torch::kCUDA, "reps must be a CUDA tensor");
	TORCH_CHECK(rep_atom_types.device().type() == torch::kCUDA, "rep_atom_types must be a CUDA tensor");
	TORCH_CHECK(indexes.device().type() == torch::kCUDA, "indexes must be a CUDA tensor");

	compute_knm_elemental_gpu(knm, sparse_points, sparse_types, reps, rep_atom_types, indexes, sigma);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("compute_kmm", &compute_kmm, "K_{MM} matrix");
	m.def("compute_knm", &compute_knm, "K_{NM} matrix");
	m.def("compute_knm_elemental_contribution", &compute_knm_elemental_contribution, "K_{NM} elemental matrix");

}
