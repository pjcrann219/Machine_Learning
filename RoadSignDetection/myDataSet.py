import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# Custom dataset class to load images and their corresponding labels
class RoadSignDataset(Dataset):
    def __init__(self, root_dir, annotations_dir, transform=None):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_paths = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.labels = self.load_labels()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def load_labels(self):
        labels = []
        for image_path in self.image_paths:
            # Extract label from XML file
            xml_path = os.path.join(self.annotations_dir, os.path.splitext(image_path)[0] + '.xml')
            label = self.extract_label_from_xml(xml_path)
            labels.append(label)
        return labels

    def extract_label_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label = root.find('object/name').text  # Replace 'class' with your XML tag for class
        return label

    def showClassStats(self):

        label_counts = Counter(self.labels)
        unique_labels = list(label_counts.keys())
        counts = list(label_counts.values())

        for label, count in zip(unique_labels, counts):
            print(f"Label: {label}, Count: {count}")

    def plotImage(self, i):
        image, label = self.__getitem__(i)
        # Display the image with label as title
        plt.figure()
        plt.title(f'Label: {label}')
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()