from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import models

# Data preprocessing (unchanged)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('./mydataset/train/images', transform=transform)
val_dataset   = datasets.ImageFolder('./mydataset/val/images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load teacher model
teacher_model = models.resnet50(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 2)
teacher_model.load_state_dict(torch.load("resnet50_teacher_model.pth"))
teacher_model.to(device)
teacher_model.eval()  # Teacher is frozen


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)   # was 8
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)   # was 16
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)  # was 32
        self.pool3 = nn.MaxPool2d(2, 2)

        self.pool4 = nn.MaxPool2d(2, 2)  # newly added

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16, 2)  # final output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
# class StudentModel(nn.Module):
#     def __init__(self):
#         super(StudentModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # Increased filters
#         self.pool1 = nn.MaxPool2d(2, 2)

#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Increased filters
#         self.pool2 = nn.MaxPool2d(2, 2)

#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Increased filters
#         self.pool3 = nn.MaxPool2d(2, 2)

#         self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # Additional layer
#         self.pool4 = nn.MaxPool2d(2, 2)

#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(128, 2)  # Output layer with increased size

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x


student_model = StudentModel().to(device)

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
    soft_labels = F.softmax(teacher_logits / temperature, dim=1)
    soft_preds = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_preds, soft_labels, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * ce_loss

# Optimizer setup
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        optimizer.zero_grad()
        student_outputs = student_model(inputs)

        loss = distillation_loss(student_outputs, teacher_outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(student_outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")

    # Validation loop
    student_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    # Save model
    torch.save(student_model.state_dict(), 'student_model_tiny.pth')

# # Save model
# torch.save(student_model.state_dict(), 'student_model_tiny.pth')
