import os

import torch


def smoothquant_prepare_quantize(self, quant_params={}):
    model = self.model_runner.model

    def update_act_scales(act_scales, x):
        # 动态统计每次输入的最大值
        hidden_dim = x.shape[-1]
        x = x.view(-1, hidden_dim).abs().detach()
        # [k]
        comming_max = torch.max(x, dim=0, keepdim=True)[0].float()

        if act_scales is None:
            act_scales = comming_max
        else:
            act_scales = torch.max(act_scales, comming_max)
        return act_scales

    from functools import partial

    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    def new_forward(input_, m, raw_forward):
        if not hasattr(m, "act_scales"):
            m.act_scales = None
        m.act_scales = update_act_scales(m.act_scales, input_)
        return raw_forward(input_)

    for name, m in model.named_modules():
        if (
            isinstance(m, QKVParallelLinear)
            or isinstance(m, RowParallelLinear)
            or isinstance(m, MergedColumnParallelLinear)
            or isinstance(m, ColumnParallelLinear)
        ):
            m.forward = partial(new_forward, m=m, raw_forward=m.forward)

def smoothquant_export_quantized_weights(self, save_path, quant_params={}):
    gb_per_file = quant_params.get("filesize_limit", None)
    smooth_alpha = quant_params.get("smooth_alpha", 0.5)
    dynamic_quant_type = quant_params.get("dynamic_quant_type", "gpu")
    assert dynamic_quant_type in ["gpu","cpu","kernel"]
    if self.rank == 0:
        print(f"set smooth_alpha={smooth_alpha}")
        print(f"use quantize weight type: {dynamic_quant_type}")
    
    import ixformer._C as ops
    def per_token_quant_8bit(weight):
        # weight: [m,k]
        dtype = weight.dtype
        i8_weight = weight
        scale = i8_weight.abs().max(dim=-1, keepdim=True)[0] / 127
        i8_weight = i8_weight / scale.to(dtype)
        i8_weight = torch.clamp(torch.round(i8_weight), -128, 127).to(torch.int8)
        return i8_weight, scale.float()

    def smooth_quant_weight_gpu_cpu(weight, act_scale, alpha=0.5, device="cpu"):
        device = torch.device("cpu") if device == "cpu" else weight.device
        ori_dtype = weight.dtype
        # [1, k]
        act_scale = act_scale.float().to(device).view(1, -1)
        weight = weight.to(device)
        # [1, k]
        weight_scale = weight.abs().max(dim=0, keepdim=True)[0].float()
        if alpha == -1:
            smooth_scales = torch.ones_like(act_scale)
        else:
            smooth_scales = act_scale.pow(alpha) / weight_scale.pow(1 - alpha).clamp(
                min=1e-5
            )
        weight = weight * smooth_scales.to(ori_dtype)
        i8_weight, weight_scales = per_token_quant_8bit(weight)
        # 为了可以使用 input * smooth_scales
        if alpha == -1:
            smooth_scales = torch.ones_like(act_scale)
        else:
            smooth_scales = weight_scale.pow(1 - alpha) / act_scale.pow(alpha).clamp(
                min=1e-5
            )
        return i8_weight, weight_scales, smooth_scales.to(ori_dtype)
    
    def smooth_quant_weight_kernel(weight, act_scale, alpha=0.5):
        output = torch.zeros_like(weight,dtype=torch.int8)
        weight_scales = torch.zeros(weight.shape[:-1],dtype=torch.float, device=weight.device)
        weight_max = torch.zeros(weight.shape[-1],dtype=torch.float, device=weight.device)
        smooth_scales = torch.zeros(weight.shape[-1],dtype=weight.dtype, device=weight.device)
        ops.infer.weight_quant_smoothquant(
            weight, act_scale, alpha, output, weight_scales, smooth_scales, weight_max
        )
        return output, weight_scales.view(-1,1), smooth_scales.view(1,-1)
    
    def smooth_quant_weight(weight, act_scale, alpha=0.5):
        if dynamic_quant_type == "kernel":
            return smooth_quant_weight_kernel(weight,act_scale,alpha)
        else:
            return smooth_quant_weight_gpu_cpu(weight,act_scale,alpha,dynamic_quant_type)

    model = self.model_runner.model

    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        get_tensor_model_parallel_world_size
    )
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.models.falcon import FalconForCausalLM

    for name, m in model.named_modules():
        if isinstance(m, VocabParallelEmbedding):
            # weight shape: [vocab_size // tp, embedding_dim]
            weight_tensor = tensor_model_parallel_all_gather(m.weight, dim=0)
            weight_tensor = weight_tensor[:m.org_vocab_size,:].contiguous()
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(weight_tensor.cpu(), requires_grad=False)
                print(f"merged: {name}, shape={m.weight.shape}")
            
        elif isinstance(m, ParallelLMHead):
            # weight shape: [vocab_size // tp, embedding_dim]
            # bias shape: [vocab_size // tp]
            if m.bias is not None:
                bias = tensor_model_parallel_all_gather(m.bias, dim=0)
                bias = bias[:m.org_vocab_size].contiguous()
                if self.is_driver_worker:
                    m.bias = torch.nn.Parameter(bias.cpu(), requires_grad=False)
            
            weight_tensor = tensor_model_parallel_all_gather(m.weight, dim=0)
            weight_tensor = weight_tensor[:m.org_vocab_size,:].contiguous()
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(weight_tensor.cpu(), requires_grad=False)
                print(f"merged: {name}, shape={m.weight.shape}")
        
        elif isinstance(m, QKVParallelLinear):
            # weight shape: [total_num_head * head_size // tp + 2 * total_num_head * head_size // tp, hidden_size]
            # bias shape: [total_num_head * head_size // tp + 2 * total_num_head * head_size // tp]
            if self.parallel_config.world_size > 1:
                total_q_hidden_size = m.total_num_heads * m.head_size
                partial_q_hidden_size = m.num_heads * m.head_size
                total_kv_hidden_size = m.total_num_kv_heads * m.head_size
                partial_kv_hidden_size = m.num_kv_heads * m.head_size
                
                if m.bias is not None:
                    # TODO do not support padding..
                    bias_tenosr = m.bias.new_zeros(total_q_hidden_size + total_kv_hidden_size * 2, m.hidden_size)
                    q_bias = bias_tenosr[:total_q_hidden_size][self.rank * partial_q_hidden_size : (self.rank + 1) * partial_q_hidden_size]
                    k_bias = bias_tenosr[total_q_hidden_size:total_q_hidden_size+total_kv_hidden_size]\
                                [self.rank * partial_kv_hidden_size : (self.rank + 1) * partial_kv_hidden_size]
                    v_bias = bias_tenosr[total_q_hidden_size+total_kv_hidden_size:]\
                                [self.rank * partial_kv_hidden_size : (self.rank + 1) * partial_kv_hidden_size]
                    
                    q_bias[:] = m.bias[:partial_q_hidden_size]
                    k_bias[:] = m.bias[partial_q_hidden_size:partial_q_hidden_size+partial_kv_hidden_size]
                    v_bias[:] = m.bias[partial_q_hidden_size+partial_kv_hidden_size:]
                    
                    bias_tensor = tensor_model_parallel_all_reduce(bias_tenosr)
                    if self.is_driver_worker:
                        m.bias = torch.nn.Parameter(bias_tensor.cpu(), requires_grad=False)

                q_tensor = m.weight.new_zeros(m.total_num_heads * m.head_size, m.weight.shape[1])
                k_tensor = m.weight.new_zeros(m.total_num_kv_heads * m.head_size, m.weight.shape[1])
                v_tensor = m.weight.new_zeros(m.total_num_kv_heads * m.head_size, m.weight.shape[1])
                
                q_in_weight = m.weight[:-m.num_kv_heads * m.head_size * 2]
                k_in_weight = m.weight[-m.num_kv_heads * m.head_size * 2:-m.num_kv_heads * m.head_size]
                v_in_weight = m.weight[-m.num_kv_heads * m.head_size:]
                
                if getattr(m,"start_idx",None) is not None:
                    start_idx = getattr(m,"start_idx")
                    weight_end_idx = m.num_heads * m.head_size if not getattr(m,"is_padding") else (m.num_heads - 1) * m.head_size
                    end_idx = start_idx + weight_end_idx
                else:
                    start_idx = self.rank * m.num_heads * m.head_size
                    weight_end_idx = m.num_heads * m.head_size
                    end_idx = start_idx + weight_end_idx
                assert q_tensor[start_idx:end_idx,:].shape == q_in_weight[:weight_end_idx, :].shape
                q_tensor[start_idx:end_idx,:] = q_in_weight[:weight_end_idx, :]
                
                if m.num_kv_head_replicas > 1:
                    if self.rank % m.num_kv_head_replicas == 0:
                        rank = self.rank // m.num_kv_head_replicas
                        k_tensor[rank * m.num_kv_heads * m.head_size:(rank+1) * m.num_kv_heads * m.head_size] = k_in_weight
                        v_tensor[rank * m.num_kv_heads * m.head_size:(rank+1) * m.num_kv_heads * m.head_size] = v_in_weight
                else:
                    k_tensor[self.rank * m.num_kv_heads * m.head_size:(self.rank+1) * m.num_kv_heads * m.head_size] = k_in_weight
                    v_tensor[self.rank * m.num_kv_heads * m.head_size:(self.rank+1) * m.num_kv_heads * m.head_size] = v_in_weight
                
                q_tensor = tensor_model_parallel_all_reduce(q_tensor)
                k_tensor = tensor_model_parallel_all_reduce(k_tensor)
                v_tensor = tensor_model_parallel_all_reduce(v_tensor)
                
                if isinstance(model, FalconForCausalLM):
                    num_query_heads_per_kv_head = (
                        m.total_num_heads // m.total_num_kv_heads
                    )
                    q_tensor = q_tensor.view(
                        m.total_num_kv_heads,
                        num_query_heads_per_kv_head,
                        m.head_size,
                        -1,
                    )
                    k_tensor = k_tensor.view(m.total_num_kv_heads, 1, m.head_size, -1)
                    v_tensor = v_tensor.view(m.total_num_kv_heads, 1, m.head_size, -1)
                    weight_tensor = torch.cat(
                        [q_tensor, k_tensor, v_tensor], dim=1
                    ).view(-1, m.hidden_size)
                else:
                    weight_tensor = torch.cat([q_tensor, k_tensor, v_tensor])
                assert (
                    weight_tensor.shape[0]
                    == total_q_hidden_size + total_kv_hidden_size * 2
                )
                assert weight_tensor.shape[1] == m.hidden_size

            else:
                weight_tensor = m.weight
                if m.bias is not None and self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
            
            if self.is_driver_worker:
                i8_weight, weight_scales, smooth_scales = smooth_quant_weight(
                    weight_tensor, m.act_scales, smooth_alpha
                )
                m.weight = torch.nn.Parameter(i8_weight.cpu(), requires_grad=False)
                m.weight_scales = torch.nn.Parameter(
                    weight_scales.cpu(), requires_grad=False
                )
                m.smooth_scales = torch.nn.Parameter(
                    smooth_scales.cpu(), requires_grad=False
                )
                print(f"Quantized: {name}")
            
        elif isinstance(m, MergedColumnParallelLinear):
            if self.parallel_config.world_size > 1:
                # weight shape: [intermediate_size // tp * 2, hidden_size]
                # bias shape: [intermediate_size // tp * 2]
                output_sizes = m.output_sizes
                output_size = sum(output_sizes)
                partial_output_sizes = [
                    i // self.parallel_config.world_size for i in output_sizes
                ]

                if m.bias is not None:
                    index_start = 0
                    partial_index_start = 0
                    bias_tenosr = m.bias.new_zeros(output_size)
                    for i in range(len(output_sizes)):
                        index_out = index_start + output_sizes[i]
                        sub_bias_tensor = bias_tenosr[index_start:index_out]
                        partial_size = partial_output_sizes[i]
                        sub_bias_tensor[self.rank * partial_size:(self.rank+1) * partial_size] = m.bias[partial_index_start:partial_index_start+partial_size]
                        
                        index_start += output_sizes[i]
                        partial_index_start += partial_size
                    bias_tenosr = tensor_model_parallel_all_reduce(bias_tenosr)
                    if self.is_driver_worker:
                        m.bias = torch.nn.Parameter(bias_tenosr, requires_grad=False)
                
                weight_tensor = m.weight.new_zeros(output_size, m.input_size)

                idx_out_start = 0
                idx_partial_satrt = 0
                for i in range(len(output_sizes)):
                    idx_out_end = idx_out_start + output_sizes[i]
                    sub_weight_tensor = weight_tensor[idx_out_start:idx_out_end]
                    partial_size = partial_output_sizes[i]
                    sub_weight_tensor[
                        self.rank * partial_size : (self.rank + 1) * partial_size
                    ] = m.weight[idx_partial_satrt : idx_partial_satrt + partial_size]

                    idx_out_start += output_sizes[i]
                    idx_partial_satrt += partial_size
                weight_tensor = tensor_model_parallel_all_reduce(weight_tensor)
            else:
                weight_tensor = m.weight
                if m.bias is not None and self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
            
            if self.is_driver_worker:
                i8_weight, weight_scales, smooth_scales = smooth_quant_weight(
                    weight_tensor, m.act_scales, smooth_alpha
                )
                m.weight = torch.nn.Parameter(i8_weight.cpu(), requires_grad=False)
                m.weight_scales = torch.nn.Parameter(
                    weight_scales.cpu(), requires_grad=False
                )
                m.smooth_scales = torch.nn.Parameter(
                    smooth_scales.cpu(), requires_grad=False
                )
                print(f"Quantized: {name}")

        elif isinstance(m, ColumnParallelLinear):
            # weight shape: [some_dim // tp, hidden_size] // for this Linear, some_dim mostly is hidden_size * 4
            # bias shape: [some_dim // tp]
            if m.bias is not None:
                bias_tenosr = tensor_model_parallel_all_gather(m.bias, dim=0)
                if self.is_driver_worker:
                    m.bias = torch.nn.Parameter(bias_tenosr.cpu(), requires_grad=False)

            weight_tensor = tensor_model_parallel_all_gather(m.weight, dim=0)

            if self.is_driver_worker:
                i8_weight, weight_scales, smooth_scales = smooth_quant_weight(
                   weight_tensor, m.act_scales, smooth_alpha
                )
                m.weight = torch.nn.Parameter(i8_weight.cpu(), requires_grad=False)
                m.weight_scales = torch.nn.Parameter(
                    weight_scales.cpu(), requires_grad=False
                )
                m.smooth_scales = torch.nn.Parameter(
                    smooth_scales.cpu(), requires_grad=False
                )
                print(f"Quantized: {name}")

        elif isinstance(m, RowParallelLinear):
            # weight shape: [hidden_size, some_dim // tp] // for this Linear, some_dim mostly is hidden_size * 4 or intermediate_size
            # bias shape: [hidden_size]
            if m.bias is not None:
                bias_tensor = tensor_model_parallel_all_gather(m.bias, dim=-1)
                if self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
            
            if getattr(m,"start_idx", None) is not None:
                start_idx = getattr(m,"start_idx")
                end_idx = start_idx + (m.input_size_per_partition if not getattr(m,"is_padding") else (m.input_size_per_partition - m.padding_size))
                weight_end_idx = m.input_size_per_partition if not getattr(m,"is_padding") else (m.input_size_per_partition - m.padding_size)
            else:
                start_idx = m.input_size_per_partition * self.rank
                end_idx = start_idx + m.input_size_per_partition
                weight_end_idx = m.input_size_per_partition
            
            act_scales = m.act_scales.new_zeros(m.input_size)
            assert act_scales[start_idx:end_idx].shape == m.act_scales.view(-1)[:weight_end_idx].shape
            act_scales[start_idx:end_idx] = m.act_scales.view(-1)[:weight_end_idx]
            act_scales = tensor_model_parallel_all_reduce(act_scales)
            m.act_scales = act_scales
            
            weight_tensor = m.weight.new_zeros(m.weight.shape[0],m.input_size)
            assert  weight_tensor[:,start_idx:end_idx].shape == m.weight[:,:weight_end_idx].shape
            weight_tensor[:,start_idx:end_idx] = m.weight[:,:weight_end_idx]
            weight_tensor = tensor_model_parallel_all_reduce(weight_tensor)
            
            if self.is_driver_worker:
                i8_weight, weight_scales, smooth_scales = smooth_quant_weight(
                    weight_tensor, m.act_scales, smooth_alpha
                )
                smooth_scales = smooth_scales.view(1,-1)
                m.weight = torch.nn.Parameter(i8_weight.cpu(), requires_grad=False)
                m.weight_scales = torch.nn.Parameter(
                    weight_scales.cpu(), requires_grad=False
                )
                m.smooth_scales = torch.nn.Parameter(
                    smooth_scales.cpu(), requires_grad=False
                )
                print(f"Quantized: {name}")
        else:
            pass
        
        torch.cuda.empty_cache()

    # save weights
    if self.is_driver_worker:
        from safetensors.torch import save_file
        
        tensors = {}
        saved = False
        count = 0
        size_in_bytes = 0
        
        tensors = {}
        for name, weight in model.named_parameters():
            if "act_scales" in name:
                continue
            # skip lm_head_weight if needed..
            if "lm_head" in name and model.config.tie_word_embeddings:
                continue
            tensors[name] = weight
            
            saved = False
            if gb_per_file is not None and size_in_bytes >= gb_per_file * 1024 * 1024 * 1024:
                weight_path = os.path.join(save_path, "model_{}.safetensors".format(str(count).zfill(6)))
                save_file(tensors, weight_path)
                print(f"The quantified weights were successfully saved in {weight_path}.")
                tensors.clear()
                saved = True
                count += 1
                size_in_bytes = 0
        
        if not saved: 
            weight_path = os.path.join(save_path, "model_{}.safetensors".format(str(count).zfill(6)))   
            save_file(tensors, weight_path)
            print(f"The quantified weights were successfully saved in {weight_path}.")