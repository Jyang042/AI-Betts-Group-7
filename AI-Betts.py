import sys

# required 3.12 version
if [sys.version_info[i] for i in range(3)][:2] != [3, 12]: 
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
from PIL import Image
import torch
# import torchvision

# Constants
PROGRAM_NAME = "BETTSAI"
DEBUG_SKIP = True
CATEGORIES = ["Glioma", "Meningioma", "Pituitary", "NoTumor"]
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAP = {
    "NoTumor": 0,
    "Glioma": 1,
    "Meningioma": 2,
    "Pituitary": 3,
}

def display_runtime(start_time: float):
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    hours, hr_rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(hr_rem, 60)
    print("[{}]: Elapsed time: {:0>2}hrs {:0>2}mins {:05.2f}secs".format(PROGRAM_NAME, int(hours),int(minutes),seconds))

def generate_dataframe(folder_path: str) -> pd.DataFrame:
    print(f"[{PROGRAM_NAME}]: Generating new dataframe from '{folder_path}' with one-hot encoding")
    dataframe = pd.DataFrame([], columns = ["Image", "Location", "Class", "Glioma", "Meningioma", "Pituitary", "NoTumor"])
    classes = os.listdir(folder_path)
    image_paths = [[image_path for image_path in os.listdir(os.path.join(folder_path, class_path))] for class_path in classes]
    
    for (i, class_name) in enumerate(classes):
        for img_path in image_paths[i]:
            # tumor classification: ["Class", "Glioma", "Meningioma", "Pituitary", "NoTumor"]
            match class_name:
                case "glioma": classification = ["Glioma", True, False, False, False]
                case "meningioma": classification = ["Meningioma", False, True, False, False]
                case "pituitary": classification = ["Pituitary", False, False, True, False]
                case "notumor": classification = ["NoTumor", False, False, False, True]
            
            # temporarily random until data can be classified and labeled
            xpos = round(random.uniform(0, 100), 2)
            ypos = round(random.uniform(0, 100), 2)

            # new data
            new_row = pd.DataFrame(
                {
                    "Image": [img_path],
                    "Location": [(xpos, ypos)],
                    "Class": [classification[0]],
                    "Glioma": [classification[1]],
                    "Meningioma": [classification[2]],
                    "Pituitary": [classification[3]],
                    "NoTumor": [classification[4]],
                }
            )

            # append parsed data
            dataframe = pd.concat([dataframe, new_row], ignore_index=True)
            dataframe["Class"] = dataframe["Class"].map(LABEL_MAP)
            
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

def visualize_dataframe(dataframe: pd.DataFrame, title: str):
    plt.figure(title)
    ax = sns.countplot(dataframe, x="Class")
    ax.set_title(title)
    ax.set_ylabel("Count")
    plt.show()
    
def visualize_dataframe_txt(dataframe: pd.DataFrame, h_bar: int, title: str):
    print(title+":", "\n"+"-"*h_bar+"\n", dataframe, "\n"+"-"*h_bar+"\n")

def visualize_data(raw_folder_path: str, amount: int = 4, image_size: tuple[int, int] = (224, 224)):
    _, axes = plt.subplots(len(CATEGORIES), amount, figsize=(10, 5), num="Data")

    for row, category in enumerate(CATEGORIES):
        category_path = os.path.join(raw_folder_path, category.lower())
        image_filenames = random.sample(os.listdir(category_path), amount)
        
        for col, image_filename in enumerate(image_filenames):
            root, ext = os.path.splitext(image_filename)
            while ext != '.jpg':
                image_filename = random.sample(os.listdir(category_path), 1)[0]
            image_path = os.path.join(category_path, image_filename)
            image = Image.open(image_path).resize(image_size)
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(f"{category}")
            
    plt.tight_layout()
    plt.show()

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

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img = Image.open(img_path).convert('RGB')  

        if self.transform:
            img = self.transform(img)
            
        return img, label

if __name__ == "__main__":
    # time program
    start_time = time.perf_counter()
    
    # title
    print("="*30, "Welcome To BETTSAI!".center(30), "="*30, sep="\n")

    # config options
    if DEBUG_SKIP:
        print(f"[{PROGRAM_NAME}]: Debug mode enabled")
    IMG_VISUALS, TEXT_VISUALS = False, False
    if not DEBUG_SKIP:
        IMG_VISUALS = input("Show data image visualizations? [y/n]: ").lower() in ["yes", "y"]
        TEXT_VISUALS = input("Show data text visualizations? [y/n]: ").lower() in ["yes", "y"]

    # file paths
    data_folder = "data"
    raw_training_folderpath = os.path.join(data_folder, "Training")
    raw_testing_folderpath = os.path.join(data_folder, "Testing")
    training_data_filepath = os.path.join(data_folder, "training.csv")
    testing_data_filepath = os.path.join(data_folder, "testing.csv")

    # load training and testing data
    training_dataframe = load_dataframe(training_data_filepath, raw_training_folderpath)
    testing_dataframe = load_dataframe(testing_data_filepath, raw_testing_folderpath)

    # visualize training and testing data
    if IMG_VISUALS:
        visualize_dataframe(training_dataframe, "Training Image Count")
        visualize_data(raw_training_folderpath, 4, (224, 224))
        visualize_dataframe(testing_dataframe, "Testing Image Count")
        visualize_data(raw_testing_folderpath, 4, (224, 224))

    if TEXT_VISUALS:
        visualize_dataframe_txt(training_dataframe, 60, "Training Data")
        visualize_dataframe_txt(testing_dataframe, 60, "Testing Data")

    # save training and testing data
    save_dataframe(training_dataframe, training_data_filepath)
    save_dataframe(testing_dataframe, testing_data_filepath)

    # program runtime
    display_runtime(start_time)
