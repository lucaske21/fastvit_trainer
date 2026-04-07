import torch
import timm
from timm.models import create_model

def create_fastvit_model(config, num_classes):
    """
    使用 timm 創建 FastViT 模型。
    """
    model = create_model(
        config['model_name'],
        pretrained=config.get('pretrained', True),
        num_classes=num_classes,
        drop_rate=config.get('drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        drop_block_rate=config.get('drop_block_rate', None),
        global_pool=config.get('gp', 'avg'),
        bn_momentum=config.get('bn_momentum', None),
        bn_eps=config.get('bn_eps', None),
        checkpoint_path=config.get('initial_checkpoint', '')
    )
    
    return model

def get_model_summary(model, input_size=(3, 224, 224)):
    """
    獲取模型參數和 FLOPs 概覽。
    """
    from torchinfo import summary
    return summary(model, input_size=(1, *input_size), device="cpu")
