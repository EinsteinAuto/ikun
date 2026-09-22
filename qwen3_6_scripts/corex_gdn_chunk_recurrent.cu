// corex_gdn_chunk_recurrent.cu — C++ GDN chunk + recurrent algorithms
//
// Extracted from: xllm_latest/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp
// These are pure PyTorch C++ implementations — no NPU/ACL/CUDA custom kernels.
// Benefit: avoids Python loop overhead in _torch_chunk_gated_delta_rule.
//
// Functions:
//   torch_chunk_gated_delta_rule(q,k,v,g,beta, chunk_size, initial_state,
//                                output_final_state, use_qk_l2norm)
//     → (core_attn_out, last_recurrent_state)
//
//   torch_recurrent_gated_delta_rule(q,k,v,g,beta, initial_state,
//                                    output_final_state, use_qk_l2norm)
//     → (core_attn_out, last_recurrent_state)

#include <torch/extension.h>
#include <optional>
#include <tuple>
#include <vector>

namespace {

torch::Tensor l2norm(const torch::Tensor& x, int64_t dim, double eps = 1e-6) {
  auto norm = torch::sqrt(torch::sum(torch::square(x), dim, true) + eps);
  return x / norm;
}

torch::Tensor repeat_tensor_heads(const torch::Tensor& tensor,
                                  int64_t target_heads,
                                  int64_t head_dim) {
  const int64_t current_heads = tensor.size(head_dim);
  if (current_heads == target_heads) {
    return tensor;
  }
  const int64_t repeats = target_heads / current_heads;
  std::vector<int64_t> view_shape = tensor.sizes().vec();
  view_shape.insert(view_shape.begin() + head_dim + 1, 1);
  std::vector<int64_t> expand_shape = view_shape;
  expand_shape[head_dim + 1] = repeats;
  std::vector<int64_t> output_shape = tensor.sizes().vec();
  output_shape[head_dim] = target_heads;
  return tensor.unsqueeze(head_dim + 1)
      .expand(expand_shape)
      .reshape(output_shape)
      .contiguous();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    c10::optional<torch::Tensor> initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  auto initial_dtype = query.dtype();

  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }

  auto to_float32_and_transpose = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_float32_and_transpose(query);
  key = to_float32_and_transpose(key);
  value = to_float32_and_transpose(value);
  beta = to_float32_and_transpose(beta);
  g = to_float32_and_transpose(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  int64_t batch_size = key.size(0);
  int64_t num_heads = key.size(1);
  int64_t sequence_length = key.size(2);
  int64_t k_head_dim = key.size(3);
  int64_t v_head_dim = value.size(3);

  float scale_val = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale_val;

  torch::Tensor core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  } else {
    last_recurrent_state =
        initial_state.value().to(value.device(), torch::kFloat32);
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    torch::Tensor q_t = query.select(2, i);
    torch::Tensor k_t = key.select(2, i);
    torch::Tensor v_t = value.select(2, i);
    torch::Tensor g_t = g.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    torch::Tensor beta_t = beta.select(2, i).unsqueeze(-1);
    last_recurrent_state = last_recurrent_state * g_t;
    torch::Tensor kv_mem =
        torch::sum(last_recurrent_state * k_t.unsqueeze(-1), -2);
    torch::Tensor delta = (v_t - kv_mem) * beta_t;
    last_recurrent_state =
        last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2);
    core_attn_out.select(2, i) =
        torch::sum(last_recurrent_state * q_t.unsqueeze(-1), -2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

std::tuple<torch::Tensor, torch::Tensor> torch_chunk_gated_delta_rule(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta,
    int64_t chunk_size,
    c10::optional<torch::Tensor> initial_state,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  auto initial_dtype = query.dtype();
  if (use_qk_l2norm_in_kernel) {
    query = l2norm(query, -1, 1e-6);
    key = l2norm(key, -1, 1e-6);
  }
  auto to_float32 = [](torch::Tensor x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };

  query = to_float32(query);
  key = to_float32(key);
  value = to_float32(value);
  beta = to_float32(beta);
  g = to_float32(g);
  const int64_t value_num_heads = value.size(1);
  query = repeat_tensor_heads(query, value_num_heads, 1);
  key = repeat_tensor_heads(key, value_num_heads, 1);

  int64_t batch_size = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t sequence_length = query.size(2);
  int64_t k_head_dim = key.size(-1);
  int64_t v_head_dim = value.size(-1);

  int64_t pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
  query = torch::nn::functional::pad(
      query, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  key = torch::nn::functional::pad(
      key, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  value = torch::nn::functional::pad(
      value, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  beta = torch::nn::functional::pad(
      beta, torch::nn::functional::PadFuncOptions({0, pad_size}));
  g = torch::nn::functional::pad(
      g, torch::nn::functional::PadFuncOptions({0, pad_size}));

  int64_t total_sequence_length = sequence_length + pad_size;
  float scale = 1.0f / std::sqrt(static_cast<float>(query.size(-1)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);
  auto reshape_to_chunks = [chunk_size](torch::Tensor x) {
    auto shape = x.sizes();
    std::vector<int64_t> new_shape = {
        shape[0], shape[1], shape[2] / chunk_size, chunk_size, shape[3]};
    return x.reshape(new_shape);
  };

  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);

  auto g_shape = g.sizes();
  std::vector<int64_t> g_new_shape = {
      g_shape[0], g_shape[1], g_shape[2] / chunk_size, chunk_size};
  g = g.reshape(g_new_shape);
  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);

  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32);
  decay_mask = decay_mask.tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1)
                   .slice(-1, 0, i)
                   .squeeze(-2)
                   .clone()
                   .contiguous();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone().contiguous();
    auto row_unsq = row.unsqueeze(-1).contiguous();
    auto row_sub_mul = (row_unsq * sub).contiguous();
    auto row_sub_sum = row_sub_mul.sum(-2).contiguous();
    auto row_final = (row + row_sub_sum).contiguous();
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }

  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, (k_beta * g.exp().unsqueeze(-1)));
  torch::Tensor last_recurrent_state;
  if (!initial_state.has_value()) {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  } else {
    last_recurrent_state = initial_state.value().to(value);
  }
  auto core_attn_out = torch::zeros_like(value);
  mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  int64_t num_chunks = total_sequence_length / chunk_size;
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(mask, 0.0);
    auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  auto core_attn_out_shape = core_attn_out.sizes();
  std::vector<int64_t> reshape_shape = {
      core_attn_out_shape[0],
      core_attn_out_shape[1],
      core_attn_out_shape[2] * core_attn_out_shape[3],
      core_attn_out_shape[4]};
  core_attn_out = core_attn_out.reshape(reshape_shape);
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_tuple(core_attn_out, last_recurrent_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_chunk_gated_delta_rule", &torch_chunk_gated_delta_rule,
        "C++ chunked gated delta rule (from xllm upstream)",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("chunk_size") = 64,
        py::arg("initial_state") = c10::optional<torch::Tensor>(),
        py::arg("output_final_state") = false,
        py::arg("use_qk_l2norm_in_kernel") = false);
  m.def("torch_recurrent_gated_delta_rule", &torch_recurrent_gated_delta_rule,
        "C++ recurrent gated delta rule (from xllm upstream)",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("initial_state") = c10::optional<torch::Tensor>(),
        py::arg("output_final_state") = false,
        py::arg("use_qk_l2norm_in_kernel") = false);
}
