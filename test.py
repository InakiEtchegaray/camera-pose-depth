import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Versión de PyTorch: {torch.__version__}")
print(f"GPU detectada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Versión de CUDA (PyTorch): {torch.version.cuda}")