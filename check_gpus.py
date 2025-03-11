import torch
import os
import torch.distributed as dist
import os
# Check total available GPUs
num_gpus = torch.cuda.device_count()
print(f"Total GPUs Available: {num_gpus}")

if num_gpus < 2:
    print("⚠️ You have less than 2 GPUs available!")
    exit()

# Set CUDA device
torch.cuda.set_device(0)  # Set to GPU 0

# Print GPU names
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Check CUDA_VISIBLE_DEVICES
visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

# Create two tensors and move them to different GPUs
try:
    tensor1 = torch.tensor([1.0, 2.0, 3.0]).cuda(0)
    tensor2 = torch.tensor([4.0, 5.0, 6.0]).cuda(1)
    print("✅ Successfully allocated tensors on two different GPUs!")

    # Try a cross-GPU operation
    result = tensor1.to(1) + tensor2
    print(f"Cross-GPU Addition Result: {result}")

except RuntimeError as e:
    print(f"❌ CUDA Error: {e}")
