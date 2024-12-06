from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import platform
import argparse
from network import Net  # Import Net from network.py

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}%')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            # Move output to same device as target for loss calculation
            if output.device != target.device:
                output = output.to(target.device)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='MNIST Training with Target Accuracy')
    parser.add_argument('--epochs', type=int, default=20,
                      help='number of epochs to train (default: 20)')
    parser.add_argument('--target-accuracy', type=float, default=99.4,
                      help='target accuracy to achieve (default: 99.4)')
    args = parser.parse_args()
    
    torch.manual_seed(1)
    device = get_device()
    
    # Data loading and augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(0,)),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1),
            shear=(-5, 5)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Using standard MNIST split (50k train/10k test)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=train_transform),
        batch_size=32,  # Reduced batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=test_transform),
        batch_size=1000, shuffle=False)
    
    model = Net().to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal trainable parameters: {total_params:,}')
    assert total_params < 20000, f'Parameter count {total_params:,} exceeds limit of 20,000'
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )
    
    best_accuracy = 0.0
    target_achieved = False
    print("\nStarting training...")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")
        print("-" * 20)
        
        # Training
        train(model, device, train_loader, optimizer, epoch)
        
        # Test accuracy (using test set as validation)
        accuracy = test(model, device, test_loader)
        
        scheduler.step()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.2f}%")
            
            if accuracy >= args.target_accuracy and not target_achieved:
                target_achieved = True
                print(f"\n🎉 Target accuracy of {args.target_accuracy}% achieved in epoch {epoch}!")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    if target_achieved:
        print(f"✨ Successfully achieved target accuracy of {args.target_accuracy}%!")
    else:
        print(f"Target accuracy of {args.target_accuracy}% not achieved. Best was {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()