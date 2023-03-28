from mmcv.utils import Registry

from .base_model import Base_model

# create a registry
MODELS = Registry('model')


# create a build function
def build_model(model_type: str, *args, **kwargs) -> Base_model:    #返回Base_model类型
    if model_type not in MODELS:
        raise KeyError(f'Unrecognized task type {model_type}')
    else:
        model_cls = MODELS.get(model_type)  #同理获取key

    #！！！转到UCGAN.py
    model = model_cls(*args, **kwargs)  #函数作用：创建模型对象
    return model
