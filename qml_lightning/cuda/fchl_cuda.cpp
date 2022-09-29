#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void getElementTypesCUDA(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor natoms, torch::Tensor species, torch::Tensor element_types);

void FCHLCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell, torch::Tensor inv_cell,
		torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3,
		float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output);

void FCHLRepresentationAndDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist,
		torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::Tensor output, torch::Tensor grad);

void FCHLDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor grad);

void FCHLBackwardsCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor grad_in, torch::Tensor grad_out);

torch::Tensor get_element_types_gpu(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor natom_counts, torch::Tensor species) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	TORCH_CHECK(charges.device().type() == torch::kCUDA, "charges must be a CUDA tensor");

	TORCH_CHECK(species.device().type() == torch::kCUDA, "species must be a CUDA tensor");

	if (coordinates.dim() == 2) {
		coordinates = coordinates.unsqueeze(0);
	}

	if (charges.dim() == 1) {
		charges = charges.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor element_types = torch::zeros( { nbatch, natoms }, options);

	getElementTypesCUDA(coordinates, charges, natom_counts, species, element_types);

	return element_types;
}

torch::Tensor get_fchl_representation(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor two_body_gridpoints, torch::Tensor three_body_gridpoints, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut) {

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

	torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);

	FCHLCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
			three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output);

	return output;

}

torch::Tensor get_fchl_derivative(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor two_body_gridpoints, torch::Tensor three_body_gridpoints, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut) {

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

	torch::Tensor output_deriv = torch::zeros( { nbatch, natoms, natoms, 3, repsize }, options);

	FCHLDerivativeCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
			three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output_deriv);

	return output_deriv;

}

torch::Tensor fchl_backwards(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor two_body_gridpoints, torch::Tensor three_body_gridpoints, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::Tensor grad_in) {

	torch::Tensor clone_coordinates;
	torch::Tensor clone_charges;
	torch::Tensor clone_element_types;

	if (coordinates.dim() == 2) { // pad a dimension

		coordinates = coordinates.unsqueeze(0);
		charges = charges.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output_deriv = torch::zeros( { nbatch, natoms, 3 }, options);

	FCHLBackwardsCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
			three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, grad_in, output_deriv);

	return output_deriv;

}

std::vector<torch::Tensor> get_fchl_and_derivative(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist,
		torch::Tensor nneighbours, torch::Tensor two_body_gridpoints, torch::Tensor three_body_gridpoints, float eta2, float eta3, float two_body_decay,
		float three_body_weight, float three_body_decay, float rcut, bool gradients) {

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

		FCHLRepresentationAndDerivativeCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours,
				two_body_gridpoints, three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output, output_deriv);

		return {output, output_deriv};
	} else {

		torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);

		FCHLCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
				three_body_gridpoints, eta2, eta3, two_body_decay, three_body_weight, three_body_decay, rcut, output);

		return {output};
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("get_element_types_gpu", &get_element_types_gpu, "computes element types");
	m.def("get_fchl_representation", &get_fchl_representation, "FCHL19 Representation");
	m.def("get_fchl_derivative", &get_fchl_derivative, "Grad FCHL19 Representation");
	m.def("get_fchl_and_derivative", &get_fchl_and_derivative, "FCHL19 Representation and Grad");
	m.def("fchl_backwards", &fchl_backwards, "FCHL19 backwards pass for autograd");

}
