import math
from random import choice
import os
from matplotlib import pyplot as plt
import cv2
import logging
import numpy as np

class Show:

    def __init__(self):
        pass

    def generate_image_path(self, image_folder):
        file = choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, file)

        img = cv2.imread(image_path)
        return img

    def render_images(self, classes, dataset_path):
        fig =  plt.figure(figsize=(10, 10))
        rows = math.ceil(len(classes) / 3)
        columns = 3
        i = 1
        for item in classes:
            image_dir_path = os.path.join(dataset_path, item)
            fig.add_subplot(rows, columns, i)
            plt.imshow(self.generate_image_path(image_dir_path))   
            i += 1

        plt.show()

    def show_classes_histogram(self, classes_images_counts, classes):
        plt.figure()
        plt.bar(classes, classes_images_counts)
        plt.show()
        logging.info("Histogram shown")

    
    def show_result_plot(self, losses, accuracies, validation_losses, validation_accuracies, EPOCHS_RANGE):
        plt.figure(figsize=(14, 10))

        plt.subplot(1, 2, 1)
        plt.title('Training and Validation Loss')
        plt.plot(EPOCHS_RANGE, losses[0], label='Train Smpl Loss')
        plt.plot(EPOCHS_RANGE, validation_losses[0], label='Val Smpl Loss', linestyle='dashed')
        plt.plot(EPOCHS_RANGE, losses[1], label='Train Reg Loss')
        plt.plot(EPOCHS_RANGE, validation_losses[1], label='Val Reg Loss', linestyle='dashed')
        plt.plot(EPOCHS_RANGE, losses[2], label='Train Dyn Loss')
        plt.plot(EPOCHS_RANGE, validation_losses[2], label='Val Dyn Loss', linestyle='dashed')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.title('Training and Validation Accuracy')
        plt.plot(EPOCHS_RANGE, accuracies[0], label='Train Smpl Acc')
        plt.plot(EPOCHS_RANGE, validation_accuracies[0], label='Val Smpl Acc', linestyle='dashed')
        plt.plot(EPOCHS_RANGE, accuracies[1], label='Train Reg Acc')
        plt.plot(EPOCHS_RANGE, validation_accuracies[1], label='Val Reg Acc', linestyle='dashed')
        plt.plot(EPOCHS_RANGE, accuracies[2], label='Train Dyn Acc')
        plt.plot(EPOCHS_RANGE, validation_accuracies[2], label='Val Dyn Acc', linestyle='dashed')
        plt.legend(loc='upper right')

        plt.show()
        logging.info("Plot shown")