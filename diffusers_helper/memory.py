# By lllyasviel


import torch

try:
    import intel_extension_for_pytorch as ipex
except:
    ipex = None
    

cpu = torch.device('cpu')
gpu = None
if torch.cuda.is_available():
    gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
elif torch.xpu.is_available():
    gpu = torch.device(f'xpu:{torch.xpu.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_gpu_free_memory_gb(device=None):
    if device is None:
        device = gpu

    if device.type == 'cuda':
        memory_stats = torch.cuda.memory_stats(device)
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    elif device.type == 'xpu':
        # xpu memory check is approx and assumes no other process is using the device
        memory_stats = torch.xpu.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        # bytes_reserved = memory_stats['reserved_bytes.all.current']
        # bytes_inactive_reserved = bytes_reserved - bytes_active
        # bytes_free_cuda, _ = torch.xpu.mem_get_info(device)  # not supported yet
        props = torch.xpu.get_device_properties(device)
        bytes_total_available = max(0, props.total_memory - bytes_active - 4 * 1024 ** 3)
    else:
        bytes_total_available = 0
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_gpu_free_memory_gb(target_device) <= preserved_memory_gb:
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()
            elif target_device.type == 'xpu':
                torch.xpu.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    if target_device.type == 'cuda':
        torch.cuda.empty_cache()
    elif target_device.type == 'xpu':
        torch.xpu.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_gpu_free_memory_gb(target_device) >= preserved_memory_gb:
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()
            elif target_device.type == 'xpu':
                torch.xpu.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    if target_device.type == 'cuda':
        torch.cuda.empty_cache()
    elif target_device.type == 'xpu':
        torch.xpu.empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
