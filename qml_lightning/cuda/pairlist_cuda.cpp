#include <torch/extension.h>
#include <iostream>

using namespace at;
using namespace std;

void getNumNeighboursCUDA(torch::Tensor coordinates, torch::Tensor num_neighbours, float rcut);
void getNeighbourListCUDA(torch::Tensor coordinates, torch::Tensor neighbour_list, float rcut);
void safeFillCUDA(torch::Tensor pairlist);

torch::Tensor get_num_neighbours_gpu(torch::Tensor coordinates, float rcut) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	int nbatch = coordinates.size(0);

	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor num_neighbours = torch::zeros( { nbatch, natoms }, options);

	getNumNeighboursCUDA(coordinates, num_neighbours, rcut);

	return num_neighbours;
}

void safe_fill_gpu(torch::Tensor pairlist) {
	/* replaces -1 entries in pairlist with an arbitrary safe atom index so 1/rij doesn't throw nans */

	TORCH_CHECK(pairlist.device().type() == torch::kCUDA, "pairlist must be a CUDA tensor");

	safeFillCUDA(pairlist);
}

torch::Tensor get_neighbour_list_gpu(torch::Tensor coordinates, int max_neighbours, float rcut) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	int nbatch = coordinates.size(0);

	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor nbh_list = torch::zeros( { nbatch, natoms, max_neighbours }, options);

	nbh_list.fill_(-1);

	getNeighbourListCUDA(coordinates, nbh_list, rcut);

	return nbh_list;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("safe_fill_gpu", &safe_fill_gpu, "");
	m.def("get_neighbour_list_gpu", &get_neighbour_list_gpu, "");
	m.def("get_num_neighbours_gpu", &get_num_neighbours_gpu, "");
}
