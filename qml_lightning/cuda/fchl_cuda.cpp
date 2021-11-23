#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void FCHLCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3,
		float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output);

void FCHLDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3,
		float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output, torch::Tensor grad);

std::vector<torch::Tensor> get_fchl(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor two_body_gridpoints,
		torch::Tensor three_body_gridpoints, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		bool gradients) {

	torch::Tensor clone_coordinates;
	torch::Tensor clone_charges;
	torch::Tensor clone_element_types;

	int nspecies = species.size(0);

	int nRs2 = two_body_gridpoints.size(0);
	int nRs3 = three_body_gridpoints.size(0);

	int repsize = nspecies * nRs2 + (nspecies * (nspecies + 1)) * nRs3;

	if (coordinates.dim() == 2) { // pad a dimension

		coordinates = coordinates.unsqueeze(0);
		charges = charges.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	if (gradients) {

		torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);
		torch::Tensor output_deriv = torch::zeros( { nbatch, natoms, natoms, 3, repsize }, options);

		FCHLDerivativeCuda(coordinates, charges, species, element_types, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
				three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output, output_deriv);

		return {output, output_deriv};
	} else {

		torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);

		FCHLCuda(coordinates, charges, species, element_types, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
				three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output);

		return {output};
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_fchl", &get_fchl, "FCHL19 Representation");
}
