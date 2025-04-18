# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define your StudentModel again
# class StudentModel(nn.Module):
#     def __init__(self):
#         super(StudentModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 28 * 28, 64)
#         self.fc2 = nn.Linear(64, 2)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = x.view(-1, 32 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Load model
# model = StudentModel()
# model.load_state_dict(torch.load("student_model.pth"))
# model.eval()

# # Dummy input (size = 224x224, 3 channels)
# dummy_input = torch.randn(1, 3, 224, 224)

# # Export to ONNX
# torch.onnx.export(
#     model,
#     dummy_input,
#     "student_model.onnx",
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
#     opset_version=11
# )

# print("✅ Exported to student_model.onnx")








# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.utils.prune as prune

# # Step 1: Define model
# class StudentModel(nn.Module):
#     def __init__(self):
#         super(StudentModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32 * 28 * 28, 64)
#         self.fc2 = nn.Linear(64, 2)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = x.view(-1, 32 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Step 2: Initialize model and apply SAME pruning structure as before loading
# model = StudentModel()

# # Apply the same pruning
# prune.random_unstructured(model.conv1, name="weight", amount=0.1)
# prune.random_unstructured(model.conv2, name="weight", amount=0.25)
# prune.random_unstructured(model.conv3, name="weight", amount=0.25)
# prune.random_unstructured(model.fc1, name="weight", amount=0.35)
# prune.random_unstructured(model.fc2, name="weight", amount=0.1)

# # Step 3: Now load pruned state_dict successfully
# model.load_state_dict(torch.load("student_model_pruned.pth", map_location="cpu"))

# # Step 4: Remove pruning reparameterizations (make it permanent)
# def remove_pruning(m):
#     for name, module in m.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             try:
#                 prune.remove(module, 'weight')
#             except ValueError:
#                 pass

# remove_pruning(model)
# model.eval()

# # Step 5: Export to ONNX
# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(model, dummy_input, "student_model_pruned.onnx",
#                   input_names=['input'], output_names=['output'],
#                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
#                   opset_version=11)

# print("✅ Successfully exported to student_model_pruned.onnx")









# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define the same student model architecture
# class StudentModel(nn.Module):
#     def __init__(self):
#         super(StudentModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(32, 2)

#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

# # Load the model and weights
# model = StudentModel()
# model.load_state_dict(torch.load("student_model_new.pth", map_location="cpu"))
# model.eval()


# dummy_input = torch.randn(1, 3, 224, 224)  # match input shape of your model
# torch.onnx.export(
#     model,
#     dummy_input,
#     "student_model_new.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#     opset_version=11
# )










import torch
import torch.nn as nn
import torch.nn.functional as F
# Define student model (updated)
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


# Load the trained model
model = StudentModel()
model.load_state_dict(torch.load("student_model_tiny.pth"))
model.eval()

# Dummy input (128x128 image, 3 channels)
dummy_input = torch.randn(1, 3, 64, 64)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "student_model_tiny.onnx",
    input_names=["input"],
    output_names=["output"],
    # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
