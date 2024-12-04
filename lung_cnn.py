
"""
Dec 4  2024



PULMONARY DISEASE CLASSIFICATION BY USING LUNG X-RAYS VIA DEEP LEARNING



Dataset: https://data.mendeley.com/datasets/9d55cttn5h/1
    Will be also shared with the repository.
    
Jagannath University
Md Alamin Talukder





    
Reference: Please refer with Oguzhan Memis, or at least Github and Dataset links provided:
    https://github.com/O-Memis
    
    
Contacts are welcomed.




CODE ORGANIZATION:
    This code is a well documented CNN application via Pytorch library.
    The codes are separated into 4 different cells with total 17 steps.
    Read the descriptions and run the codes cell by cell. 
    You can also run the whole code at once, if you want.
    
    The steps are as follows:
        
        1) Inspect the image attributes
        2) Pytorch CNN model:
                             # 1. Processing Unit Selection
                             # 2. Define and change Hyperparameters
                             # 3. Define a data preprocessing tool
                             # 4. Dataset loading via applying transformations
                             # 5. Data split ratios
                             # 6. Train-test splitting
                             # 7. Data loader functions
                             # 8. Model Definitions
                             # 9. The Loss function
                             # 10. The optimization selection
                             # 11. Initialize metric tracking lists
                             # 12. Training and testing loops
                             # 13. Visualization of metrics
    
        3) Save the model
        4) Load and use the model

"""




#%% 1) Inspect the image attributes


import os
import cv2

# Directory containing your images
image_dir = 'data/Viral_Pneumonia'

# List to store image sizes and channels
image_info = []

# Iterate through each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image extensions if needed
        # Load the image using OpenCV
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Get the size and channels
            height, width, channels = img.shape  # (height, width, channels)

            # Append the information to the list
            image_info.append((filename, (width, height), channels))
        else:
            print(f"Could not read image: {img_path}")

# Print the image information
for info in image_info:
    print(f"Image: {info[0]}, Size: {info[1]}, Channels: {info[2]}")






#%% 2) Pytorch CNN model



import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Processing Unit Selection
print("CUDA Availability:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing Device: {device}")



# 2. Define and change Hyperparameters here
lr = 0.001 # learning rate
batch = 32 # batch size, chunks of data loaded for forward pass each time, reduces calculation
epochs = 30 # number of total iterations. Each time all the data is forward passed, is an epoch



# 3. Define a data preprocessing tool
transform = transforms.Compose([transforms.Resize((296, 296)),transforms.ToTensor(),])



# 4. Dataset loading via applying transformations
lungdataset = datasets.ImageFolder('data', transform=transform) # It is helpful on identification of labels, automatically from folder names.
# One-hot encoding is not needed in this case.


# 5. Data split ratios
dataset_size = len(lungdataset)
train_size = int(0.7 * dataset_size)  # accepts only integer numbers, which corresponds the real size
test_size = dataset_size - train_size  # remaining



# 6. Train-test splitting. It allows preventing mixture and leakage between train and test data. It splits in the same way in each run.
train_data , test_data = random_split(lungdataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))



# 7. Data loader functions, to mix in each epoch, and load them in batches
train_loading = DataLoader(train_data, batch_size= batch, shuffle=True)
test_loading = DataLoader(test_data, batch_size= batch, shuffle=True)   # It prevents the dependency on data order






# 8. MODEL DEFINITION : 
# Two methods can be used 1- nn.Sequential()  or more traditional  2- Custom class definition

lungcnn = nn.Sequential(
    
    nn.Conv2d(3, 32, 3, padding=1),     # channels of the image, number of generated feature maps, size of kernel, padding optional.
    
    nn.BatchNorm2d(32),                 # applying normalization to values in feature maps
    
    nn.LeakyReLU(0.1),                  # another hyperparameter here, in the activation function
    
    nn.MaxPool2d(2,2),                  # halving the image size
    
    nn.Conv2d(32, 64, 3, padding=1),    # now takes previous feature maps as input
    
    nn.BatchNorm2d(64),                 # should match with the feature map number
    
    nn.LeakyReLU(0.1),
    
    nn.MaxPool2d(2,2),
    
    
    nn.Flatten(),                       # now the convolutional layers ended, linear layers started which takes 1D input
    
    nn.Linear(64*74*74, 256),           # input size is feature maps*image size (maxpooled)
    
    nn.LeakyReLU(0.1),
    
    nn.Dropout(0.5),                    # optional regularization tool
    
    nn.Linear(256, 128),
    
    nn.LeakyReLU(0.1),
    
    nn.Dropout(0.2),
    
    nn.Linear(128, 3)                   # if you are not using Cross-Entropy loss, you should add a Softmax layer
    
    ).to(device)                        # moved to run in GPU


# 9. The Loss function
cost = nn.CrossEntropyLoss()            # combines Softmax for multi-label classification



# 10. The optimization algorithm, to update the defined model parameters. Can be further arranged
optimization = optim.Adam(lungcnn.parameters(), lr= lr)




# 11. Initialize metric tracking lists, stores during epochs
train_losses = []      
test_losses = []       
train_accuracies = []  
test_accuracies = []   



# 12. TRAINING AND TESTING STAGES. First loop is counting for epochs. 

for epoch in range(epochs):  
    
    lungcnn.train()  # Set model into training mode (enables dropout, batch norm)
    y_true_train = []  
    y_pred_train = []  # List to store predicted labels during training
    
    running_loss = 0.0   # Initialize loss value


    # 12.1. Training loop for batches in 1 epoch
    
    for inputs, labels in train_loading:  # Load the training data in batches, with labels
        
        inputs, labels = inputs.to(device), labels.to(device)  # GPU
        
        optimization.zero_grad()      # Reset gradients
        
        outputs = lungcnn(inputs)     # Forward pass: get model predictions
        
        loss = cost(outputs, labels)  # Calculate loss by using labels
        
        loss.backward()               # Backpropagation on FINAL loss function
       
        optimization.step()           # Update model weights
        
        
        running_loss += loss.item()   # Accumulate (sum) all loss values
        
        _, predicted = torch.max(outputs.data, 1)  # Get predicted classes via training
        # It collects the max value to get highest probability label prediction. The second term is important as "predicted"
        
        # Convert predictions and labels to CPU numpy arrays for sklearn metrics
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())
        
        # Training loop ended


    
    # Calculate training metrics for each epoch completed
    running_loss = running_loss / len(train_loading)  # Average loss by dividing it to number of batches
    
    
    # Calculate and store the metrics by using running loss value
    train_losses.append(running_loss)
    train_accuracies.append(accuracy_score(y_true_train, y_pred_train) * 100)  # Percentage of correct predictions


    
    # 12.2. Testing Phase
    
    lungcnn.eval()  # Set model to evaluation mode 
    
    y_true_test = []  
    y_pred_test = []  # List to store predicted labels during validation
    
    test_loss = 0.0   # Initialize validation loss accumulator (with batches)
    
    
    with torch.no_grad():  # Disable gradients
    
        for inputs, labels in test_loading:  # This time, it is test loader
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = lungcnn(inputs)  # Get model predictions
            loss = cost(outputs, labels)  
            
            test_loss += loss.item()  # Accumulate in batches
            
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
            
            # Testing loop ended
    
    
    # Calculate test metrics for each epoch
    test_loss = test_loss / len(test_loading)  # Average test loss
    test_losses.append(test_loss)
    test_accuracies.append(accuracy_score(y_true_test, y_pred_test) * 100)
    
    print("Epoch: " + str(epoch))
    print("Test loss: " + str(test_loss))
    print("Test accuracy: " + str(accuracy_score(y_true_test, y_pred_test)))


# All loops ended for epochs.





# 13. Visualization Section

# Confusion Matrix
plt.figure(figsize=(11, 9))  
cm = confusion_matrix(y_true_test, y_pred_test)  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# Training Curves
plt.figure(figsize=(27, 9))  

# Loss plot
plt.subplot(1, 2, 1)  # First subplot
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)  # Second subplot
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()




#%% 3) Save the model


modelpath = './lung_cnn.pth'

torch.save(lungcnn.state_dict(), modelpath)




#%% 4) Load and use the model


# Load the saved model
lungcnn.load_state_dict(torch.load(modelpath))
lungcnn.eval()  


# Further validation can be done by using a different, unseen subset of the initial dataset.

y_true_val = []
y_pred_val = []

with torch.no_grad():
    for inputs, labels in test_loading:  # Using existing test loader
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = lungcnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        y_true_val.extend(labels.cpu().numpy())
        y_pred_val.extend(predicted.cpu().numpy())


# Calculate metrics

plt.figure(figsize=(11, 9))
cm = confusion_matrix(y_true_val, y_pred_val)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()



accuracy = accuracy_score(y_true_val, y_pred_val)
precision = precision_score(y_true_val, y_pred_val, average='weighted')
f1 = f1_score(y_true_val, y_pred_val, average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test F1 Score: {f1:.4f}')



    
