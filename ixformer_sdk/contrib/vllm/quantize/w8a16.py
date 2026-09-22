import os

import torch

def w8a16_prepare_quantize(self, quant_params={}):
    # We need do nothing in here
    pass


def w8a16_export_quantized_weights(self, save_path, quant_params={}):
    gb_per_file = quant_params.get("filesize_limit", None)
    int8_min = -127

    def w8a16_quantization(weight):
        # all weights should be [output,input], otherwise, we may get an wrong weight and scale...
        scale = torch.abs(weight).max(dim=-1)[0] / 127.0
        int8_weight = torch.clamp(weight / scale.view(-1,1),min=int8_min,max=127).to(torch.int8).contiguous()
        scale = scale.view(1,-1).contiguous()
        return int8_weight, scale


    model = self.model_runner.model

    from vllm.distributed.communication_op import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
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
                
                weight_tensor = m.weight.new_zeros(
                    total_q_hidden_size + total_kv_hidden_size * 2, m.hidden_size
                )

                q_tensor = weight_tensor[:total_q_hidden_size, :]
                q_tensor = q_tensor[self.rank * partial_q_hidden_size : (self.rank + 1) * partial_q_hidden_size]

                k_tensor = weight_tensor[total_q_hidden_size : total_q_hidden_size + total_kv_hidden_size]
                k_tensor = k_tensor[self.rank * partial_kv_hidden_size : (self.rank + 1) * partial_kv_hidden_size]

                v_tensor = weight_tensor[total_q_hidden_size + total_kv_hidden_size :]
                v_tensor = v_tensor[self.rank * partial_kv_hidden_size : (self.rank + 1) * partial_kv_hidden_size]

                q_tensor[:, :] = m.weight[: partial_q_hidden_size, :]
                k_tensor[:, :] = m.weight[partial_q_hidden_size : partial_q_hidden_size + partial_kv_hidden_size, :]
                v_tensor[:, :] = m.weight[partial_q_hidden_size + partial_kv_hidden_size : , :]

                weight_tensor = tensor_model_parallel_all_reduce(weight_tensor)
            else:
                weight_tensor = m.weight
                if m.bias is not None and self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
                    
            int8_weight, weight_scales = w8a16_quantization(weight_tensor)
            
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(int8_weight.cpu(), requires_grad=False)
                m.scales = torch.nn.Parameter(weight_scales.cpu(), requires_grad=False)
                print(f"Quantized: {name}")

        elif isinstance(m, MergedColumnParallelLinear):
            # weight shape: [intermediate_size // tp * 2, hidden_size]
            # bias shape: [intermediate_size // tp * 2]
            if self.parallel_config.world_size > 1:
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

                index_start = 0
                partial_index_start = 0
                for i in range(len(output_sizes)):
                    index_out = index_start + output_sizes[i]
                    sub_weight_tensor = weight_tensor[index_start:index_out]
                    partial_size = partial_output_sizes[i]
                    sub_weight_tensor[self.rank * partial_size : (self.rank + 1) * partial_size] = m.weight[partial_index_start : partial_index_start + partial_size]

                    index_start += output_sizes[i]
                    partial_index_start += partial_size
                weight_tensor = tensor_model_parallel_all_reduce(weight_tensor)
            else:
                weight_tensor = m.weight
                if m.bias is not None and self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
                
            int8_weight, weight_scales = w8a16_quantization(weight_tensor)
            
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(int8_weight.cpu(), requires_grad=False)
                m.scales = torch.nn.Parameter(weight_scales.cpu(), requires_grad=False)
                print(f"Quantized: {name}")
        
        elif isinstance(m, ColumnParallelLinear):
            # weight shape: [some_dim // tp, hidden_size] // for this Linear, some_dim mostly is hidden_size * 4
            # bias shape: [some_dim // tp]
            if m.bias is not None:
                bias_tenosr = tensor_model_parallel_all_gather(m.bias, dim=0)
                if self.is_driver_worker:
                    m.bias = torch.nn.Parameter(bias_tenosr.cpu(), requires_grad=False)
            
            weight_tensor = tensor_model_parallel_all_gather(m.weight, dim=0)
            
            int8_weight, weight_scales = w8a16_quantization(weight_tensor)
            
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(int8_weight.cpu(), requires_grad=False)
                m.scales = torch.nn.Parameter(weight_scales.cpu(), requires_grad=False)
                print(f"Quantized: {name}")
        
        elif isinstance(m, RowParallelLinear):
            # weight shape: [hidden_size, some_dim // tp] // for this Linear, some_dim mostly is hidden_size * 4 or intermediate_size
            # bias shape: [hidden_size]
            if m.bias is not None:
                bias_tensor = tensor_model_parallel_all_gather(m.bias, dim=-1)
                if self.is_driver_worker:
                    m.bias = torch.nn.Parameter(m.bias.cpu(), requires_grad=False)
            
            weight_tensor = tensor_model_parallel_all_gather(m.weight, dim=-1)
            int8_weight, weight_scales = w8a16_quantization(weight_tensor)
            
            if self.is_driver_worker:
                m.weight = torch.nn.Parameter(int8_weight.cpu(), requires_grad=False)
                m.scales = torch.nn.Parameter(weight_scales.cpu(), requires_grad=False)
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
        
        for name, weight in model.named_parameters():
            if "lm_head" in name and model.config.tie_word_embeddings:
                continue
            size_in_bytes += weight.numel() * weight.element_size()
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
