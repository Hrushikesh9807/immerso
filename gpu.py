import torch

print(f"--- PyTorch GPU Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    current_device_id = torch.cuda.current_device()
    print(f"Current GPU ID: {current_device_id}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(current_device_id)}")
    print("\nDetails for all detected GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # You can add more properties like:
        # print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
        # print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("No CUDA-enabled GPU found or PyTorch was not built with CUDA support.")
    print("If you have an NVIDIA GPU, ensure drivers and CUDA toolkit are correctly installed,")
    print("and that you installed the CUDA-enabled version of PyTorch from https://pytorch.org/")