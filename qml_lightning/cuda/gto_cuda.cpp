#include <torch/extension.h>
#include<iostream>

using namespace at;
using namespace std;

void getElementTypesGPU(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types);

void getElementTypeBoundsGPU(torch::Tensor element_types, torch::Tensor species, torch::Tensor element_starts, torch::Tensor nelements);

void elementalGTOGPUSharedMem_float(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor mbodylist, torch::Tensor gto_components, torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, float eta,
		int lmax, float rcut, torch::Tensor gto_output);

void elementalGTOGPUSharedMemDerivative_float(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor mbodylist, torch::Tensor gto_components, torch::Tensor gto_powers, torch::Tensor orbital_weights, torch::Tensor gridpoints, float eta,
		int lmax, float rcut, torch::Tensor gto_output, torch::Tensor gto_output_derivative);

torch::Tensor get_element_types_gpu(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	TORCH_CHECK(charges.device().type() == torch::kCUDA, "charges must be a CUDA tensor");

	TORCH_CHECK(species.device().type() == torch::kCUDA, "species must be a CUDA tensor");

	if (coordinates.dim() == 2) { // pad a dimension so elementalGTOGPUSharedMem still works

		coordinates = coordinates.unsqueeze(0);
		charges = charges.unsqueeze(0);

	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor element_types = torch::zeros( { nbatch, natoms }, options);

	getElementTypesGPU(coordinates, charges, species, element_types);

	return element_types;
}

std::vector<torch::Tensor> get_element_type_bounds_gpu(torch::Tensor element_types, torch::Tensor species) {

	/**
	 *
	 *
	 * **/

	int nbatch = element_types.size(0);
	int natoms = element_types.size(1);
	int nspecies = species.size(0);

	TORCH_CHECK(element_types.device().type() == torch::kCUDA, "element_tyoes must be a CUDA tensor");

	TORCH_CHECK(species.device().type() == torch::kCUDA, "species must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor element_starts = torch::zeros( { nbatch, nspecies }, options);
	torch::Tensor nelements = torch::zeros( { nbatch, nspecies }, options);

	getElementTypeBoundsGPU(element_types, species, element_starts, nelements);

	return {element_starts, nelements};
}

std::vector<torch::Tensor> elemental_gto_gpu_shared(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor mbodylist, torch::Tensor gto_components, torch::Tensor orbital_weights, torch::Tensor gto_powers, torch::Tensor gridpoints, float eta,
		int lmax, float rcut, bool gradients) {

	/** ElementalGTO representation GPU wrapper
	 *
	 * coordinates: [nbatch, natoms, 3]
	 * charges: [nbatch, natoms]
	 * element_types: [nbatch, natoms]
	 * mbodylist: [nspecies, nspecies]
	 * gto_components: [norbs, 3]
	 * orbital_weights: [norbs]
	 * gto_powers: [norbs]
	 * gridpoints: [ngaussians]
	 *
	 * **/

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	TORCH_CHECK(charges.device().type() == torch::kCUDA, "charges must be a CUDA tensor");

	TORCH_CHECK(element_types.device().type() == torch::kCUDA, "element_types must be a CUDA tensor");

	TORCH_CHECK(mbodylist.device().type() == torch::kCUDA, "mbodylist must be a CUDA tensor");

	TORCH_CHECK(species.device().type() == torch::kCUDA, "species must be a CUDA tensor");

	TORCH_CHECK(gto_components.device().type() == torch::kCUDA, "gto_components must be a CUDA tensor");

	TORCH_CHECK(orbital_weights.device().type() == torch::kCUDA, "orbital_weights must be a CUDA tensor");

	TORCH_CHECK(gto_powers.device().type() == torch::kCUDA, "gto_powers must be a CUDA tensor");

	TORCH_CHECK(gridpoints.device().type() == torch::kCUDA, "gridpoints must be a CUDA tensor");

	torch::Tensor clone_coordinates;
	torch::Tensor clone_charges;
	torch::Tensor clone_element_types;

	int nspecies = mbodylist.size(0);

	int ngaussians = gridpoints.size(0);

	int nmbody = int(float((nspecies + 1) / 2.0) * nspecies);
	int repsize = nmbody * (lmax + 1) * ngaussians;

	if (coordinates.dim() == 2) { // pad a dimension so elementalGTOGPUSharedMem still works

		coordinates = coordinates.unsqueeze(0);
		charges = charges.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	if (gradients) {

		torch::Tensor gto_output = torch::zeros( { nbatch, natoms, repsize }, options);
		torch::Tensor gto_output_derivative = torch::zeros( { nbatch, natoms, natoms, 3, repsize }, options);

		elementalGTOGPUSharedMemDerivative_float(coordinates, charges, species, element_types, mbodylist, gto_components, gto_powers, orbital_weights,
				gridpoints, eta, lmax, rcut, gto_output, gto_output_derivative);

		return {gto_output, gto_output_derivative};
	} else {

		torch::Tensor gto_output = torch::zeros( { nbatch, natoms, repsize }, options);

		elementalGTOGPUSharedMem_float(coordinates, charges, species, element_types, mbodylist, gto_components, gto_powers, orbital_weights, gridpoints, eta,
				lmax, rcut, gto_output);

		return {gto_output};
	}
}

int main() {
	torch::Tensor tensor = torch::rand( { 2, 3 });
	std::cout << tensor << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("elemental_gto_gpu_shared", &elemental_gto_gpu_shared, "Elemental GTO Representation in Shared Memory");
	m.def("get_element_types_gpu", &get_element_types_gpu, "returns atomic species according to torch::Tensor species");
	m.def("get_element_type_bounds_gpu", &get_element_type_bounds_gpu, "species start indexes and num_atoms");
}
