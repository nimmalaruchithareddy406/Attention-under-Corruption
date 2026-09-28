import time
import torch
from torchvision import datasets, transforms
from corruptions import CIFAR10CCorruption

def test_speed():
    tf = transforms.Compose([
        CIFAR10CCorruption(p=1.0),
        transforms.ToTensor()
    ])
    ds = datasets.CIFAR10(root='C:/cifar_data', train=True, download=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    
    start = time.time()
    for i, (x, y) in enumerate(loader):
        if i >= 10: break # Test 10 batches
    
    end = time.time()
    print(f"Time for 10 corrupted batches: {end - start:.2f}s")
    print(f"Estimated time per epoch: {(end - start) * (50000/1280):.2f}s")

if __name__ == "__main__":
    test_speed()
