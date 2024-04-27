import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import ResNet34_Weights


class DomainNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the DomainNetDataset.

        Args:
            root_dir (str): The root directory path of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the dataset using ImageFolder
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        image, label = self.dataset[idx]
        return image, label


# Define the transformations to be applied to the images
transform = transforms.Compose([
    # transforms.RandomRotation(degrees=15),  # Randomly rotate the image by +/-15 degrees
    # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally with a probability of 0.5
    # transforms.RandomVerticalFlip(),  # Randomly flip the image vertically with a probability of 0.5
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Apply color jittering
    # transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 0.1
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Apply Gaussian blur with a random sigma in 0.1 and 2.0
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create instances of the DomainNetDataset for each domain
real_train_dataset = DomainNetDataset(root_dir='data/real_train', transform=transform)
real_test_dataset = DomainNetDataset(root_dir='data/real_test', transform=transform)
sketch_train_dataset = DomainNetDataset(root_dir='data/sketch_train', transform=transform)
sketch_test_dataset = DomainNetDataset(root_dir='data/sketch_test', transform=transform)

# Create DataLoader instances for each dataset
real_train_dataloader = DataLoader(real_train_dataset, batch_size=64, shuffle=True)
real_test_dataloader = DataLoader(real_test_dataset, batch_size=64, shuffle=False)
sketch_train_dataloader = DataLoader(sketch_train_dataset, batch_size=64, shuffle=True)
sketch_test_dataloader = DataLoader(sketch_test_dataset, batch_size=64, shuffle=False)

# Load the pretrained ResNet-34 model
resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

# Replace the final layer
num_classes = 10  # the number of classes
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)

# freeze the model's parameters
for param in resnet34.parameters():
    param.requires_grad = False

# do not freeze the final layer
resnet34.fc.weight.requires_grad = True
resnet34.fc.bias.requires_grad = True

# deploy to device
# for windows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for mac
# device = torch.device("mps")

resnet34 = resnet34.to(device)

# Train Loop
def train(dataloader, model, loss_fn, optimizer, device, epoch):
    # model.train()
    running_loss = 0.0
    total_loss = 0.0
    # Get a batch of training data from the DataLoader
    for batch, data in enumerate(dataloader):
        # Every data instance is an image + label pair
        img, label = data

        # Transfer data to target device
        img = img.to(device)
        label = label.to(device)

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Compute prediction for this batch
        logit = model(img)

        # compute the loss and its gradients
        loss = loss_fn(logit, label)
        # Backpropagation
        loss.backward()

        # update the parameters according to gradients
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        total_loss += loss.item()

        # report every 100 iterations
        if batch % 100 == 99:
            print(f'epoch {epoch+1} loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    return total_loss / (batch+1)


# Evaluation Loop
def test(dataloader, model, loss_fn, device):
    # Get number of batches
    num_batches = len(dataloader)

    test_loss, correct, total = 0, 0, 0

    # Context-manager that disabled gradient calculation.
    with torch.no_grad():
        for data in dataloader:
            # Every data instance is an image + label pair
            img, label = data

            # Transfer data to target device
            img = img.to(device)
            label = label.to(device)

            # Compute prediction for this batch
            logit = model(img)

            # compute the loss
            test_loss += loss_fn(logit, label).item()

            # Calculate argmax: Any maximum logit as the predicted label
            pred = logit.argmax(dim=1)
            # record correct predictions
            correct += (pred == label).type(torch.float).sum().item()
            total += label.size(0)

    # Gather data and report
    test_loss /= num_batches
    accuracy = correct / total
    print("Test Error: \n Accuracy: {:.2f}, Avg loss: {:.4f} \n".format(100 * accuracy, test_loss))

    return test_loss, accuracy

# hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# loss function and optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet34.fc.parameters(), lr=learning_rate)
print("Hyperparameters , criterion and optimizer created.")

# train model
train_losses = []
for epoch in range(num_epochs):
    print(f"Training model with epoch {epoch}")
    train_loss = train(real_train_dataloader, resnet34, criterion, optimizer, device, epoch)
    train_losses.append(train_loss)

# evaluate the model on test
test_loss, test_acc = test(real_test_dataloader, resnet34, criterion, device)
print(f'Test accuracy: {test_acc:.4f}')

# plot the loss training curve
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


