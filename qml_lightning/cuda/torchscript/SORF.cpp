#include <torch/script.h>
#include <torch/all.h>

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

using namespace at;
using namespace std;
using namespace std::chrono;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

void hadamard_gpu(torch::Tensor input, torch::Tensor dmatrix, torch::Tensor output, float normalisation, const int ntransforms);
void hadamard_backwards_gpu(torch::Tensor input, torch::Tensor dmatrix, torch::Tensor output, float normalisation, const int ntransforms);

void cos_features_gpu(torch::Tensor coeffs, torch::Tensor b, torch::Tensor batch_indexes, torch::Tensor output);
void cos_derivative_features_gpu(torch::Tensor grads, torch::Tensor coeffs, torch::Tensor b, torch::Tensor batch_indexes, torch::Tensor output);

torch::Tensor hadamard_transform_gpu(torch::Tensor input, torch::Tensor dmatrix, float normalisation, const int ntransforms) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(dmatrix.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), dmatrix.size(1), input.size(1) }, options);

	hadamard_gpu(input, dmatrix, output, normalisation, ntransforms);

	return output;
}

torch::Tensor hadamard_transform_backwards_gpu(torch::Tensor input, torch::Tensor dmatrix, float normalisation, const int ntransforms) {

	TORCH_CHECK(input.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(dmatrix.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { input.size(0), input.size(2) }, options);

	hadamard_backwards_gpu(input, dmatrix, output, normalisation, ntransforms);

	return output;
}

torch::Tensor CosFeaturesCUDA(torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { nmol, coeffs.size(1) }, options);

	cos_features_gpu(coeffs, b, batch_indexes, output);

	return output;
}

torch::Tensor CosDerivativeFeaturesCUDA(torch::Tensor grads, torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes) {

	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA);

	torch::Tensor output = torch::zeros( { coeffs.size(0), coeffs.size(1) }, options);

	cos_derivative_features_gpu(grads, coeffs, b, batch_indexes, output);

	return output;
}

class SORF: public torch::autograd::Function<SORF> {
public:
	static variable_list forward(AutogradContext *ctx, Variable u, Variable d, Variable coeff_normalisation, Variable ntransforms) {

		TORCH_CHECK(u.device().type() == torch::kCUDA, "u must be a CUDA tensor");
		TORCH_CHECK(d.device().type() == torch::kCUDA, "dmatrix must be a CUDA tensor");

		variable_list saved_input = { d, coeff_normalisation, ntransforms };

		ctx->save_for_backward(saved_input);

		return {hadamard_transform_gpu(u, d, coeff_normalisation.item<float>(), ntransforms.item<int>())};

	}

	static variable_list backward(AutogradContext *ctx, variable_list grad_output) {

		variable_list saved_vars = ctx->get_saved_variables();

		auto d = saved_vars[0];
		auto coeff_normalisation = saved_vars[1];
		auto ntransforms = saved_vars[2];

		at::Tensor undef;

		return {hadamard_transform_backwards_gpu(grad_output[0], d, coeff_normalisation.item<float>(), ntransforms.item<int>()), undef, undef, undef};

	}
};

class CosFeatures: public torch::autograd::Function<CosFeatures> {
public:
	static variable_list forward(AutogradContext *ctx, Variable coeffs, Variable b, Variable nmol, Variable batch_indexes) {

		TORCH_CHECK(coeffs.device().type() == torch::kCUDA, "coeffs must be a CUDA tensor");
		TORCH_CHECK(b.device().type() == torch::kCUDA, "b must be a CUDA tensor");
		TORCH_CHECK(batch_indexes.device().type() == torch::kCUDA, "batch_indexes must be a CUDA tensor");

		variable_list saved_input = { coeffs, b, nmol, batch_indexes };

		ctx->save_for_backward(saved_input);

		//CosFeaturesCUDA(torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes)
		return {CosFeaturesCUDA( coeffs, b, nmol.item<int>(), batch_indexes)};

	}

	static variable_list backward(AutogradContext *ctx, variable_list grad_output) {

		variable_list saved_vars = ctx->get_saved_variables();

		auto coeffs = saved_vars[0];
		auto b = saved_vars[1];
		auto nmol = saved_vars[2];
		auto batch_indexes = saved_vars[3];

		at::Tensor undef;

		//torch::Tensor CosDerivativeFeaturesCUDA(torch::Tensor grads, torch::Tensor coeffs, torch::Tensor b, int nmol, torch::Tensor batch_indexes)
		return {CosDerivativeFeaturesCUDA(grad_output[0], coeffs,b, nmol.item<int>(), batch_indexes), undef, undef, undef,undef};

	}
};

Tensor get_sorf(const Tensor &u, const Tensor &d, const Tensor &coeff_normalisation, const Tensor &ntransforms) {
	return SORF::apply(u, d, coeff_normalisation, ntransforms)[0];
}

Tensor get_cos_features(const Tensor &coeffs, const Tensor &b, const Tensor &nmol, const Tensor &batch_indexes) {
	return CosFeatures::apply(coeffs, b, nmol, batch_indexes)[0];
}

TORCH_LIBRARY(qml_lightning_sorf, m)
{
	m.def("get_SORF_coefficients", get_sorf);
	m.def("get_cos_features", get_cos_features);
}
