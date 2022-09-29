#include <torch/script.h>
#include <torch/all.h>

#include <iostream>
#include <memory>
#include <vector>

using namespace at;
using namespace std;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

void getElementTypesCUDA(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor natoms, torch::Tensor species, torch::Tensor element_types);

void getNumNeighboursCUDA(torch::Tensor coordinates, torch::Tensor natoms, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor num_neighbours);

void getNeighbourListCUDA(torch::Tensor coordinates, torch::Tensor natoms, float rcut, torch::Tensor lattice_vecs, torch::Tensor inv_lattice_vecs,
		torch::Tensor neighbour_list);

void FCHLCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell, torch::Tensor inv_cell,
		torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3,
		float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output);

void FCHLBackwardsCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell,
		torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours,
		torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut,
		torch::Tensor grad_in, torch::Tensor grad_out);

void FCHLRepresentationAndDerivativeCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types,
		torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist,
		torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3, float eta2, float eta3, float two_body_decay, float three_body_weight,
		float three_body_decay, float rcut, torch::Tensor output, torch::Tensor grad);

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

torch::Tensor get_num_neighbours_gpu(torch::Tensor coordinates, torch::Tensor natoms, torch::Tensor rcut,
		torch::Tensor lattice_vecs = torch::empty( { 0, 3, 3 }, torch::kCUDA), torch::Tensor inv_lattice_vecs = torch::empty( { 0, 3, 3 }, torch::kCUDA)) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	if (coordinates.dim() == 2) {
		coordinates = coordinates.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);

	int max_natoms = natoms.max().item<int>();

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor num_neighbours = torch::zeros( { nbatch, max_natoms }, options);

	getNumNeighboursCUDA(coordinates, natoms, rcut.item<float>(), lattice_vecs, inv_lattice_vecs, num_neighbours);

	return num_neighbours;
}

torch::Tensor get_neighbour_list_gpu(torch::Tensor coordinates, torch::Tensor natoms, torch::Tensor max_neighbours, torch::Tensor rcut,
		torch::Tensor lattice_vecs = torch::empty( { 0, 3, 3 }, torch::kCUDA), torch::Tensor inv_lattice_vecs = torch::empty( { 0, 3, 3 }, torch::kCUDA)) {

	TORCH_CHECK(coordinates.device().type() == torch::kCUDA, "coordinates must be a CUDA tensor");

	if (coordinates.dim() == 2) {
		coordinates = coordinates.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);

	int max_natoms = natoms.max().item<int>();

	int max_neigh = max_neighbours.item<int>();

	auto options = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor nbh_list = torch::zeros( { nbatch, max_natoms, max_neigh }, options);

	nbh_list.fill_(-1);

	getNeighbourListCUDA(coordinates, natoms, rcut.item<float>(), lattice_vecs, inv_lattice_vecs, nbh_list);

	return nbh_list;
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
	}

	if (charges.dim() == 1) {
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

std::vector<torch::Tensor> get_fchl_and_derivative(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor blockAtomIDs,
		torch::Tensor blockMolIDs, torch::Tensor atom_counts, torch::Tensor cell, torch::Tensor inv_cell, torch::Tensor two_body_gridpoints,
		torch::Tensor three_body_gridpoints, torch::Tensor eta2, torch::Tensor eta3, torch::Tensor two_body_decay, torch::Tensor three_body_weight,
		torch::Tensor three_body_decay, torch::Tensor rcut, bool gradients) {

	torch::Tensor clone_coordinates;
	torch::Tensor clone_charges;
	torch::Tensor clone_element_types;

	int nspecies = species.size(0);

	int nRs2 = two_body_gridpoints.size(0);
	int nRs3 = three_body_gridpoints.size(0);

	int repsize = nspecies * nRs2 + (nspecies * (nspecies + 1)) * nRs3;

	if (coordinates.dim() == 2) { // pad a dimension
		coordinates = coordinates.unsqueeze(0);
	}

	if (charges.dim() == 1) {
		charges = charges.unsqueeze(0);
	}

	int nbatch = coordinates.size(0);
	int natoms = coordinates.size(1);

	torch::Tensor nneighbours = get_num_neighbours_gpu(coordinates, atom_counts, rcut, cell, inv_cell);

	torch::Tensor max_neighbours = nneighbours.max();

	torch::Tensor neighbourlist = get_neighbour_list_gpu(coordinates, atom_counts, max_neighbours, rcut, cell, inv_cell);
	torch::Tensor element_types = get_element_types_gpu(coordinates, charges, atom_counts, species);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	if (gradients) {

		torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);
		torch::Tensor output_deriv = torch::zeros( { nbatch, natoms, natoms, 3, repsize }, options);

		FCHLRepresentationAndDerivativeCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours,
				two_body_gridpoints, three_body_gridpoints, eta2.item<float>(), eta3.item<float>(), two_body_decay.item<float>(),
				three_body_weight.item<float>(), three_body_decay.item<float>(), rcut.item<float>(), output, output_deriv);

		return {output, output_deriv};
	} else {

		torch::Tensor output = torch::zeros( { nbatch, natoms, repsize }, options);

		FCHLCuda(coordinates, charges, species, element_types, cell, inv_cell, blockAtomIDs, blockMolIDs, neighbourlist, nneighbours, two_body_gridpoints,
				three_body_gridpoints, eta2.item<float>(), eta3.item<float>(), two_body_decay.item<float>(), three_body_weight.item<float>(),
				three_body_decay.item<float>(), rcut.item<float>(), output);

		return {output};
	}
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
	}

	if (charges.dim() == 1) {
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

class FCHL19: public torch::autograd::Function<FCHL19> {
public:

	static variable_list forward(AutogradContext *ctx, Variable X, Variable Z, Variable species, Variable atomIDs, Variable molIDs, Variable atom_counts,
			Variable cell, Variable inv_cell, Variable Rs2, Variable Rs3, Variable eta2, Variable eta3, Variable two_body_decay, Variable three_body_weight,
			Variable three_body_decay, Variable rcut) {

		//void FCHLCuda(torch::Tensor coordinates, torch::Tensor charges, torch::Tensor species, torch::Tensor element_types, torch::Tensor cell, torch::Tensor inv_cell,
		// torch::Tensor blockAtomIDs, torch::Tensor blockMolIDs, torch::Tensor neighbourlist, torch::Tensor nneighbours, torch::Tensor Rs2, torch::Tensor Rs3,
		// float eta2, float eta3, float two_body_decay, float three_body_weight, float three_body_decay, float rcut, torch::Tensor output);
		TORCH_CHECK(X.device().type() == torch::kCUDA, "charges must be a CUDA tensor");
		TORCH_CHECK(Z.device().type() == torch::kCUDA, "charges must be a CUDA tensor");
		TORCH_CHECK(species.device().type() == torch::kCUDA, "species must be a CUDA tensor");
		TORCH_CHECK(atomIDs.device().type() == torch::kCUDA, "atomIDs must be a CUDA tensor");
		TORCH_CHECK(molIDs.device().type() == torch::kCUDA, "molIDs must be a CUDA tensor");
		TORCH_CHECK(atom_counts.device().type() == torch::kCUDA, "atom_counts must be a CUDA tensor");
		TORCH_CHECK(cell.device().type() == torch::kCUDA, "cell must be a CUDA tensor");
		TORCH_CHECK(inv_cell.device().type() == torch::kCUDA, "inv_cell must be a CUDA tensor");

		torch::Tensor nneighbours = get_num_neighbours_gpu(X, atom_counts, rcut, cell, inv_cell);

		torch::Tensor max_neighbours = nneighbours.max();

		torch::Tensor neighbourlist = get_neighbour_list_gpu(X, atom_counts, max_neighbours, rcut, cell, inv_cell);
		torch::Tensor element_types = get_element_types_gpu(X, Z, atom_counts, species);

		variable_list saved_input = { X, Z, species, atomIDs, molIDs, element_types, neighbourlist, nneighbours, cell, inv_cell, Rs2, Rs3, eta2, eta3,
				two_body_decay, three_body_weight, three_body_decay, rcut };

		ctx->save_for_backward(saved_input);

		torch::Tensor output = get_fchl_representation(X, Z, species, element_types, cell, inv_cell, atomIDs, molIDs, neighbourlist, nneighbours, Rs2, Rs3,
				eta2.item<float>(), eta3.item<float>(), two_body_decay.item<float>(), three_body_weight.item<float>(), three_body_decay.item<float>(),
				rcut.item<float>());

		return {output};
	}

	static variable_list backward(AutogradContext *ctx, variable_list grad_output) {

		variable_list saved_vars = ctx->get_saved_variables();

		auto gradX = grad_output[0];

		auto X = saved_vars[0];
		auto Z = saved_vars[1];
		auto species = saved_vars[2];

		auto atomIDs = saved_vars[3];
		auto molIDs = saved_vars[4];
		auto element_types = saved_vars[5];

		auto neighbourlist = saved_vars[6];
		auto nneighbours = saved_vars[7];

		auto cell = saved_vars[8];
		auto inv_cell = saved_vars[9];

		auto Rs2 = saved_vars[10];
		auto Rs3 = saved_vars[11];
		auto eta2 = saved_vars[12];
		auto eta3 = saved_vars[13];
		auto two_body_decay = saved_vars[14];
		auto three_body_weight = saved_vars[15];
		auto three_body_decay = saved_vars[16];
		auto rcut = saved_vars[17];

		at::Tensor undef;

		torch::Tensor grad_out = fchl_backwards(X, Z, species, element_types, cell, inv_cell, atomIDs, molIDs, neighbourlist, nneighbours, Rs2, Rs3,
				eta2.item<float>(), eta3.item<float>(), two_body_decay.item<float>(), three_body_weight.item<float>(), three_body_decay.item<float>(),
				rcut.item<float>(), gradX);

		return {grad_out, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef, undef};
	}
};

Tensor fchl_forwards(const Tensor &x, const Tensor &Z, const Tensor &species, const Tensor &atomIDs, const Tensor &molIDs, const Tensor &atom_counts,
		const Tensor &cell, const Tensor &inv_cell, const Tensor &Rs2, const Tensor &Rs3, const Tensor &eta2, const Tensor &eta3, const Tensor &two_body_decay,
		const Tensor &three_body_weight, const Tensor &three_body_decay, const Tensor &rcut) {

	return FCHL19::apply(x, Z, species, atomIDs, molIDs, atom_counts, cell, inv_cell, Rs2, Rs3, eta2, eta3, two_body_decay, three_body_weight, three_body_decay,
			rcut)[0];
}

TORCH_LIBRARY(qml_lightning_fchl, m) {
	m.def("fchl_forwards", fchl_forwards);
	m.def("get_fchl_and_derivative", get_fchl_and_derivative);

	m.def("get_element_types", get_element_types_gpu);
	m.def("get_neighbour_list", get_neighbour_list_gpu);
	m.def("get_num_neighbours", get_num_neighbours_gpu);
}
