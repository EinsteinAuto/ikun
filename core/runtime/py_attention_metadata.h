/* Adapted from xLLM commit 78aa2a85 (PR #2258).
   Adds dp_token_counts / dp_is_decode fields to PyAttentionMetadataView
   so the Python attention backend can partition KV cache by DP group.

   Original: xllm/core/runtime/py_attention_metadata.h
   Scope:    Qwen3.5 data-parallel support in project_6.
==============================================================================*/

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

/* Forward declarations — project_6 keeps these in its own layer namespace. */
namespace project6::layer {
struct AttentionMetadata;
struct ExpandedDecodeMetadata;
}  // namespace project6::layer

namespace project6 {

struct ModelInputParams;

void register_attention_metadata_views(pybind11::module_& module);

class PyExpandedDecodeMetadataView final {
 public:
  explicit PyExpandedDecodeMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata);

  bool enabled() const;
  pybind11::object kv_seq_lens() const;
  pybind11::object block_table() const;
  pybind11::object paged_kv_indptr() const;
  pybind11::object paged_kv_indices() const;
  pybind11::object paged_kv_last_page_len() const;
  pybind11::object paged_attention_tiling_data() const;
  pybind11::object kv_seq_lens_host() const;
  const std::vector<int32_t>& kv_seq_lens_host_values() const;

 private:
  const layer::ExpandedDecodeMetadata& metadata() const;

  std::shared_ptr<layer::AttentionMetadata> metadata_;
};

class PyAttentionMetadataView final {
 public:
  explicit PyAttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata);
  PyAttentionMetadataView(std::shared_ptr<layer::AttentionMetadata> metadata,
                          const ModelInputParams& params);

  const torch::Tensor& slot_mapping() const;
  const torch::Tensor& paged_kv_indptr() const;
  const torch::Tensor& paged_kv_indices() const;
  const torch::Tensor& paged_kv_last_page_len() const;
  pybind11::object qo_indptr() const;
  pybind11::object q_cu_seq_lens() const;
  pybind11::object kv_cu_seq_lens() const;
  pybind11::object kv_seq_lens_host() const;
  const std::vector<int32_t>& kv_seq_lens_host_values() const;
  pybind11::object q_seq_lens_host() const;
  pybind11::object block_table() const;
  pybind11::object kv_seq_lens() const;
  pybind11::object linear_state_indices() const;
  pybind11::object has_initial_state() const;

  /* ---- DP fields (added by PR #2258) ---------------------------------- */
  const std::vector<int32_t>& dp_token_counts() const;
  const std::vector<int32_t>& dp_is_decode() const;
  /* --------------------------------------------------------------------- */

  pybind11::object q_seq_lens() const;
  PyExpandedDecodeMetadataView expanded_decode_metadata() const;
  bool is_prefill() const;
  bool is_chunked_prefill() const;

 private:
  static torch::Tensor make_host_int32_view(
      const std::shared_ptr<layer::AttentionMetadata>& metadata,
      std::vector<int32_t>& host_vec);
  static pybind11::object optional_tensor(const torch::Tensor& tensor);

  std::shared_ptr<layer::AttentionMetadata> metadata_;
  torch::Tensor kv_seq_lens_host_;
  torch::Tensor q_seq_lens_host_;
  torch::Tensor linear_state_indices_;

  /* ---- DP fields (added by PR #2258) ---------------------------------- */
  std::vector<int32_t> dp_token_counts_;
  std::vector<int32_t> dp_is_decode_;
  /* --------------------------------------------------------------------- */
};

}  // namespace project6
