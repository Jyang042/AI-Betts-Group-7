import sys

if [sys.version_info[i] for i in range(3)][:2] != [3, 12]: # required 3.12 version
    raise Exception(f"Python 3.12 is required (Current is {[sys.version_info[i] for i in range(3)]})")

import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import imutils
import numpy as np

#pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# Constants
PROGRAM_NAME = "BETTSAI"
DEBUG_SKIP = False
CATEGORIES = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]

def display_runtime(start_time: float):
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    hours, hr_rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(hr_rem, 60)
    print("[{}]: Elapsed time: {:0>2}hrs {:0>2}mins {:05.2f}secs".format(PROGRAM_NAME, int(hours),int(minutes),seconds))

def generate_dataframe(folder_path: str) -> pd.DataFrame:
    print(f"[{PROGRAM_NAME}]: Generating new dataframe from '{folder_path}' with one-hot encoding")
    dataframe = pd.DataFrame([], columns = ["Image", "Glioma", "Meningioma", "Pituitary", "NoTumor"])
    classes = os.listdir(folder_path)
    image_paths = [[image_path for image_path in os.listdir(os.path.join(folder_path, class_path))] for class_path in classes]
    
    for (i, class_name) in enumerate(classes):
        for img_path in image_paths[i]:
            # tumor classification: ["Glioma", "Meningioma", "Pituitary", "NoTumor"]
            match class_name:
                case "glioma": classification = [True, False, False, False]
                case "meningioma": classification = [False, True, False, False]
                case "pituitary": classification = [False, False, True, False]
                case "notumor": classification = [False, False, False, True]

            # new data
            new_row = pd.DataFrame(
                {
                    "Image": [img_path],
                    "Glioma": [classification[0]],
                    "Meningioma": [classification[1]],
                    "Pituitary": [classification[2]],
                    "NoTumor": [classification[3]],
                }
            )

            # append parsed data
            dataframe = pd.concat([dataframe, new_row], ignore_index=True)
            
    return dataframe

def save_dataframe(dataframe: pd.DataFrame, filepath: str):
    # Save to csv files
    print(f"[{PROGRAM_NAME}]: Saving dataframe to '{filepath}'")
    dataframe.to_csv(filepath, index=False)

def load_dataframe(filepath: str, raw_folderpath: str) -> pd.DataFrame:
    if os.path.isfile(filepath):
        print(f"[{PROGRAM_NAME}]: Loading CSV file '{filepath}'")
        return pd.read_csv(filepath)
    else:
        print(f"[{PROGRAM_NAME}]: Dataframe save file not found: '{filepath}'")
        return generate_dataframe(raw_folderpath)

def visualize_dataframe(dataframe: pd.DataFrame, column: str, title: str):
    sns.countplot(dataframe, x="Class")
    plt.figure(title, figsize=(10, 5))
    x_axis = sns.countplot(data=dataframe, y=dataframe[column])
    x_axis.bar_label(x_axis.containers[0])
    plt.title(title, fontsize=20)
    plt.show()
    
def visualize_dataframe_txt(dataframe: pd.DataFrame, h_bar: int, title: str):
    print(title+":", "\n"+"-"*h_bar+"\n", dataframe, "\n"+"-"*h_bar+"\n")

def preprocess_image(image_path: str, image_size: tuple[int, int]): # (128, 128)
    # Validate image path
    if not os.path.exists(image_path):
        print(f"Image not found: '{image_path}'")
        return None # Skip missing images

    # Read image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to create binary mask
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours and select the largest one
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        print(f"No contours found, skipping image: '{image_path}'")
        return None # Skip if no contours

    c = max(contours, key=cv2.contourArea)

    # Get extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = 0  # Adjust if needed
    cropped_img = image[extTop[1] - ADD_PIXELS: extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS: extRight[0] + ADD_PIXELS].copy()

    # Apply sharpening filter
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(cropped_img, -1, sharpen_kernel)

    # Resize to standard size
    resized = cv2.resize(sharpened, image_size)

    # Min-max normalization (scale 0-1)
    normalized = resized.astype(np.float32) / 255.0

    return normalized

class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, image_folder, image_size=(128, 128)):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.image_size = image_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_folder, row['Image'])
        image = preprocess_image(image_path, self.image_size)

        if image is None:
            # Return a blank image if preprocessing fails
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)

        image = np.transpose(image, (2, 0, 1))  # HWC to CHW for PyTorch
        image_tensor = torch.tensor(image, dtype=torch.float32)

        if row["Glioma"] == 1:
            label = 0
        elif row["Meningioma"] == 1:
            label = 1
        elif row["Pituitary"] == 1:
            label = 2
        elif row["NoTumor"] == 1:
            label = 3
        else:
            raise ValueError("Invalid label in row")

        label = torch.tensor(label, dtype=torch.long)

        return image_tensor, label

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # (in_channels, out_channels, kernel_size)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by 2x

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),  # (input_features, output_features)
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 outputs for 4 tumor types
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def predict_single_image(model, image_path, image_size=(128, 128)):
    model.eval()  # Set model to evaluation mode
    image = preprocess_image(image_path, image_size)

    if image is None:
        print(f"Skipping prediction: Image '{image_path}' is invalid.")
        return None

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        
        # Get the class index with the highest score (logit)
        _, predicted_class_idx = torch.max(output, 1)
        
    return predicted_class_idx.item()  # Return the index of the predicted class

def decode_prediction(prediction, threshold=0.5):
    classes = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]
    predicted_classes = [classes[i] for i, pred in enumerate(prediction) if pred >= threshold]
    return predicted_classes if predicted_classes else ["Uncertain"]

# Function to predict and display a grid of randomly selected test images
def predict_and_display_images(model, dataframe, image_folder, sample_size=5, image_size=(128, 128), threshold=0.5):
    model.eval()
    
    # Sample image rows without replacement
    sampled_rows = dataframe.sample(n=sample_size, replace=False)
    
    image_paths = [
        os.path.join(image_folder, row['Image']) for _, row in sampled_rows.iterrows()
    ]

    num_images = len(image_paths)
    cols = min(num_images, 4)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, (index, row) in enumerate(sampled_rows.iterrows()):
        image_path = os.path.join(image_folder, row['Image'])
        prediction = predict_single_image(model, image_path, image_size)
        if prediction is None:
            continue
        
        predicted_class = CATEGORIES[prediction]
        
        # Load original image for display
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_rgb)
        plt.axis("off")
        
        # Get true label from the dataframe row
        true_label = CATEGORIES[np.argmax([
            row["Glioma"], row["Meningioma"], row["Pituitary"], row["NoTumor"]
        ])]
        
        #display truth label to match with prediction value
        plt.title(
            f"{os.path.basename(image_path)}\nPredicted: {predicted_class}\nTrue: {true_label}",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # time program
    start_time = time.perf_counter()

    # config options
    if DEBUG_SKIP:
        print(f"[{PROGRAM_NAME}]: Debug mode enabled")
    IMG_VISUALS, TEXT_VISUALS = False, False
    if not DEBUG_SKIP:
        #IMG_VISUALS = input("Show data image visualizations? [y/n]: ").lower() in ["yes", "y"]
        TEXT_VISUALS = input("Show data text visualizations? [y/n]: ").lower() in ["yes", "y"]

    # file paths
    data_folder = "data"
    raw_training_folderpath = os.path.join(data_folder, "Training")
    raw_testing_folderpath = os.path.join(data_folder, "Testing")
    training_data_filepath = os.path.join(data_folder, "training.csv")
    testing_data_filepath = os.path.join(data_folder, "testing.csv")
    model_save_path = "bettai_cnn_model.pth"
    
    # load training and testing data
    training_dataframe = load_dataframe(training_data_filepath, raw_training_folderpath)
    testing_dataframe = load_dataframe(testing_data_filepath, raw_testing_folderpath)

    # visualize training and testing data
    if IMG_VISUALS:
        visualize_dataframe(training_dataframe, "Class", "Training Image count")
        visualize_dataframe(testing_dataframe, "Class", "Testing Image count")

    if TEXT_VISUALS:
        visualize_dataframe_txt(training_dataframe, 60, "Training Data")
        visualize_dataframe_txt(testing_dataframe, 60, "Testing Data")

    # save training and testing data
    save_dataframe(training_dataframe, training_data_filepath)
    save_dataframe(testing_dataframe, testing_data_filepath)
    
    # Prepare datasets
    train_dataset = BrainTumorDataset(training_dataframe, raw_training_folderpath)
    test_dataset = BrainTumorDataset(testing_dataframe, raw_testing_folderpath)

    # Split the original training dataset into training and validation (e.g., 80% train, 20% val)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize CNN
    model = BrainTumorCNN()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Check if model already exists
    if os.path.exists(model_save_path):
        # Load the trained model
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        print(f"[{PROGRAM_NAME}]: Loaded model and training stats from '{model_save_path}'")
    else:
        # Training loop
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        EPOCHS = 10
        print(f"[{PROGRAM_NAME}]: Starting CNN training")
        
        for epoch in range(EPOCHS):
            # --- Training ---
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # --- Validation ---
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_loss.item()

                    _, val_predicted = torch.max(val_outputs, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_loss_epoch = val_running_loss / len(val_loader)
            val_acc_epoch = 100 * val_correct / val_total
            val_losses.append(val_loss_epoch)
            val_accuracies.append(val_acc_epoch)

            # Print results for this epoch
            print(f"Epoch [{epoch+1}/{EPOCHS}], "
                f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, "
                f"Val Loss: {val_loss_epoch:.4f}, Val Accuracy: {val_acc_epoch:.2f}%")
     
                 
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, model_save_path)

    SAMPLE_IMAGES = 5
    print(f"[{PROGRAM_NAME}]: Displaying predictions for {SAMPLE_IMAGES} random test images...\n")
    predict_and_display_images(model, testing_dataframe, raw_testing_folderpath, sample_size=SAMPLE_IMAGES) 
        
    #plot loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
        
    # program runtime
    display_runtime(start_time)