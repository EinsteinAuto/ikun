import torch
import torch.nn as nn
from abc import ABC
from ixformer.train.speedformer.model_replacer_mapping import ModelMapping

# 外部接口
class SpeedFormer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.replacer = None


    def accelerate(self, model):
        if model.config.model_type in ModelMapping:
            self.replacer = ModelMapping[model.config.model_type]()
            accelerate_model = self.replacer.accelerate(model)
        else:
            Warning(f"Warning: model '{model.config.model_type}' is not supported now.")
            accelerate_model = model

        return accelerate_model
    
    
    def post_process(self, model):
        self.replacer.post_process(model)
