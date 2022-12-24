import os
import cv2
import logging
import pandas
import tensorflow as tf

class Data:

    def __init__(self):
        self.CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.CLASSES_COUNT = len(self.CLASSES)
        self.DATASET_PATH = r"D:/lessons/Machine Learning/LR2/notMNIST_large/notMNIST_large"
        self.DATA_COLUMN_NAME = 'data'
        self.LABELS_COLUMN_NAME = 'labels'
        self.HASHED_DATA_COLUMN_NAME = 'data_bytes'
        self.BALANCE_BORDER = 0.85
        self.TRAIN_SIZE = 200000
        self.VALIDATION_SIZE = 10000
        self.TEST_SIZE = 19000
        self.BATCH_SIZE = 48
        self.EPOCHS = 30
        self.EPOCHS_RANGE = range(self.EPOCHS)


    def get_class_data(self, folder_path):
        result_data = list()
        files = os.listdir(folder_path)
        for file in files:
            image_path = os.path.join(folder_path, file)
            img = cv2.imread(image_path)
            if img is not None:
                result_data.append(img.reshape(-1))

        return result_data


    def get_classes_images_counts(self, data_frame):
        classes_images_counts = list()
        for class_index in range(len(self.CLASSES)):
            labels = data_frame[self.LABELS_COLUMN_NAME]
            class_rows = data_frame[labels == class_index]
            class_count = len(class_rows)

            classes_images_counts.append(class_count)
            logging.info(f"Class {self.CLASSES[class_index]} contains {class_count} images")

        return classes_images_counts

    
    def check_classes_balance(self, data_frame):
        classes_images_counts = self.get_classes_images_counts(data_frame)

        max_images_count = max(classes_images_counts)
        avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
        balance_percent = avg_images_count / max_images_count

    
        logging.info(f"Balance: {balance_percent:.3f}")
        if balance_percent > self.BALANCE_BORDER:
            logging.info("Classes are balanced")
        else:
            logging.info("Classes are not balanced")

        return classes_images_counts


    def create_data_frame(self):
        data = list()
        labels = list()
        for class_item in self.CLASSES:
            class_folder_path = os.path.join(self.DATASET_PATH, class_item)
            class_data = self.get_class_data(class_folder_path)

            data.extend(class_data)
            labels.extend([self.CLASSES.index(class_item) for _ in range(len(class_data))])

        data_frame = pandas.DataFrame({self.DATA_COLUMN_NAME: data, self.LABELS_COLUMN_NAME: labels})
        logging.info("Data frame is created")

        return data_frame


    def remove_duplicates(self, data):
        data_bytes = [item.tobytes() for item in data[self.DATA_COLUMN_NAME]]
        data[self.HASHED_DATA_COLUMN_NAME] = data_bytes
        data.sort_values(self.HASHED_DATA_COLUMN_NAME, inplace=True)
        data.drop_duplicates(subset=self.HASHED_DATA_COLUMN_NAME, keep='first', inplace=True)
        data.pop(self.HASHED_DATA_COLUMN_NAME)
        logging.info("Duplicates removed")

        return data

    def shuffle_data(self, data):
        data_shuffled = data.sample(frac=1, random_state=42)
        logging.info("Data shuffled")

        return data_shuffled

    def split_dataset_into_subsamples(self, data_frame):
        data = list(data_frame[self.DATA_COLUMN_NAME].values)
        labels = list(data_frame[self.LABELS_COLUMN_NAME].values)

        data_dataset = tf.data.Dataset.from_tensor_slices(data)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        dataset = tf.data.Dataset.zip((data_dataset, labels_dataset))

        train_dataset = dataset.take(self.TRAIN_SIZE).batch(self.BATCH_SIZE)
        validation_dataset = dataset.skip(self.TRAIN_SIZE).take(self.VALIDATION_SIZE).batch(self.BATCH_SIZE)
        test_dataset = dataset.skip(self.TRAIN_SIZE + self.VALIDATION_SIZE).take(self.TEST_SIZE).batch(self.BATCH_SIZE)
        logging.info("Data split")
        return train_dataset, validation_dataset, test_dataset


    def get_statistics(self, model, train_dataset, validation_dataset, test_dataset, with_optimization=False):
        if with_optimization:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = tf.keras.optimizers.experimental.SGD()

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        model_history = model.fit(
            x=train_dataset,
            validation_data=validation_dataset,
            epochs=self.EPOCHS,
            verbose=1
        )
        loss, accuracy = model.evaluate(test_dataset)
        logging.info(f"Model: {accuracy=}, {loss=}")

        accuracy = model_history.history['accuracy']
        validation_accuracy = model_history.history['val_accuracy']
        loss = model_history.history['loss']
        validation_loss = model_history.history['val_loss']

        return loss, accuracy, validation_loss, validation_accuracy


    def get_neural_network_statistics(self, train_dataset, validation_dataset, test_dataset):
        losses = list()
        accuracies = list()
        validation_losses = list()
        validation_accuracies = list()

        train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        simple_model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1 / 255.),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.CLASSES_COUNT)
        ])

        regularized_model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1 / 255.),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.CLASSES_COUNT)
        ])

        dynamic_model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1 / 255.),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.CLASSES_COUNT)
        ])

        dynamic_model_stats = self.get_statistics(
            dynamic_model, train_dataset, validation_dataset, test_dataset, with_optimization=True
        )

        regularized_model_stats = self.get_statistics(
            regularized_model, train_dataset, validation_dataset, test_dataset
        )

        simple_model_stats = self.get_statistics(
            simple_model, train_dataset, validation_dataset, test_dataset
        )

        losses.extend((simple_model_stats[0], regularized_model_stats[0], dynamic_model_stats[0]))
        accuracies.extend((simple_model_stats[1], regularized_model_stats[1], dynamic_model_stats[1]))
        validation_losses.extend((simple_model_stats[2], regularized_model_stats[2], dynamic_model_stats[2]))
        validation_accuracies.extend((simple_model_stats[3], regularized_model_stats[3], dynamic_model_stats[3]))

        return losses, accuracies, validation_losses, validation_accuracies
