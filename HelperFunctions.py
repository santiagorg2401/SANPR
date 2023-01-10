import os
import random
import zipfile
import datetime
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

"""
This library is based on helper_functions.py by Daniel Bourke for the ZTM: Deep Learning course and its original file avaialble in:
https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/helper_functions.py
"""


class ImportHelperFunctions:
    def __init__(self):
        pass

    # Import and load data functions.

    def unzip_data(self, filename):
        """
        Unzips filename into the current working directory.

        Args:
        filename (str): a filepath to a target zip folder to be unzipped.
        """
        zip_ref = zipfile.ZipFile(filename, "r")
        zip_ref.extractall()
        zip_ref.close()

    def walk_through_dir(self, dir_path):
        """
        Walks through dir_path returning its contents.

        Args:
        dir_path (str): target directory

        Returns:
        A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
        """
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(
                f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    def view_random_image(self, target_dir, target_class, i=1):
        """
        Get a random image path, read it and plot it.
        target_dir: String. It contains the target directory where the iamges are stored.
        target_class: String. The class name of the image to plot.

        i: Integer. Number of images to retrieve.
        """
        # Setup the targer directory.
        target_folder = target_dir + target_class

        imgs = []

        for j in range(0, i):
            # Get a random image path.
            random_image = random.sample(os.listdir(target_folder), 1)
            print(random_image)
            # Read in the image and plot it.
            img = mpimg.imread(target_folder + "/" + random_image[0])
            fig, (ax) = plt.subplots()
            ax.imshow(img)
            ax.set_title(target_class)
            ax.set_axis_off()
            print(f"Image shape: {img.shape}")
            imgs.append(img)

        plt.show()
        return img

    def load_and_prep_image(self, img_size, filename, scale=True):
        """
        Reads in an image from filename, turns it into a tensor and reshapes into
        (224, 224, 3).

        Parameters
        ----------
        filename (str): string filename of target image
        img_shape (int): size to resize target image to, default 224
        scale (bool): whether to scale pixel values to range(0, 1), default True
        """
        # Read in the image
        img = tf.io.read_file(filename)
        # Decode it into a tensor
        img = tf.image.decode_jpeg(img)
        # Resize the image
        img = tf.image.resize(img, size=img_size)
        if scale:
            # Rescale the image (get all values between 0 and 1)
            return img/255.
        else:
            return img


class ModelHelperFunctions(ImportHelperFunctions):
    def __init__(self, img_shape, model, class_names, test_data=None):
        self.__img_shape = img_shape
        self.__model = model
        self.__class_names = class_names

        if test_data != None:
            self.__test_data = test_data

            self.__y_true = []
            for images, labels in test_data.unbatch():
                # Check whether or not labels is Int or OneHot encoded.
                if (labels.numpy().shape == ()):
                    self.__y_true.append(labels.numpy())
                else:
                    self.__y_true.append(labels.numpy().argmax())

            self.__y_probs = self.__model.predict(test_data)
            self.__y_preds = self.__y_probs.argmax(axis=1)

    # Make predictions.
    def pred_and_plot(self, filename, scale=True):
        """
        Imports a foreign image located at filename, makes a prediction on it with
        a trained model and plots the image with the predicted class as the title.
        """
        # Import the target image and preprocess it
        img = self.load_and_prep_image(img_size=self.__img_shape,
                                       filename=filename, scale=scale)

        # Make a prediction
        pred = self.__model.predict(tf.expand_dims(img, axis=0))

        # Get the predicted class
        if len(pred[0]) > 1:  # check for multi-class
            # if more than one output, take the max
            pred_class = self.__class_names[pred.argmax()]
        else:
            # if only one output, round
            pred_class = self.__class_names[int(tf.round(pred)[0][0])]

        # Plot the image and predicted class
        plt.figure()
        if scale == False:
            plt.imshow(img/255.)
        else:
            plt.imshow(img)

        plt.title(f"Prediction: {pred_class}")
        plt.axis(False)
        plt.show()

    def pred_from_batch(self, figsize=(10, 10), numImgs=9):
        """
        Grabs a certain amount of images from test data set and plots its predictions.
        Args:
        figsize: Tuple containing size of plot.
        numImgs: Integer containig number of images to predict and plot.
        """
        plt.figure(figsize=figsize)
        i = 0
        numRows = int(numImgs / 3) + (numImgs % 3 > 0)
        for image, label in self.__test_data.unbatch().take(numImgs):
            y_prob = self.__model.predict(tf.expand_dims(image, axis=0))
            y_pred = self.__class_names[y_prob.argmax()]

            if (label.numpy().shape == ()):
                y_true = label.numpy()
            else:
                y_true = label.numpy().argmax()

            y_true = self.__class_names[y_true]

            if y_pred == y_true:
                color = 'g'
            else:
                color = 'r'

            ax = plt.subplot(numRows, 3, i + 1)
            plt.imshow(image/255.)
            plt.title(f"Prediction: {y_pred}, Actual: {y_true}", c=color)
            plt.axis("off")
            i += 1

    # Evaluate the model.

    def plot_loss_curves(self, history):
        """
        Returns separate loss curves for training and validation metrics.

        Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
        """
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))

        # Plot loss
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

    def compare_historys(self, original_history, new_history, initial_epochs=5):
        """
        Compares two TensorFlow model History objects.

        Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after original_history)
        initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
        """

        # Get original history measurements
        acc = original_history.history["accuracy"]
        loss = original_history.history["loss"]

        val_acc = original_history.history["val_accuracy"]
        val_loss = original_history.history["val_loss"]

        # Combine original history with new history
        total_acc = acc + new_history.history["accuracy"]
        total_loss = loss + new_history.history["loss"]

        total_val_acc = val_acc + new_history.history["val_accuracy"]
        total_val_loss = val_loss + new_history.history["val_loss"]

        # Make plots
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(total_acc, label='Training Accuracy')
        plt.plot(total_val_acc, label='Validation Accuracy')
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(total_loss, label='Training Loss')
        plt.plot(total_val_loss, label='Validation Loss')
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

    def calculate_results(self):
        """
        Calculates model accuracy, precision, recall and f1 score of a binary classification model.

        Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array

        Returns a dictionary of accuracy, precision, recall, f1-score.
        """
        # Calculate model accuracy
        model_accuracy = accuracy_score(self.__y_true, self.__y_preds) * 100
        # Calculate model precision, recall and f1 score using "weighted average
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
            self.__y_true, self.__y_preds, average="weighted")
        model_results = {"accuracy": model_accuracy,
                         "precision": model_precision,
                         "recall": model_recall,
                         "f1": model_f1}

        return pd.DataFrame([model_results]).transpose()

    def make_classification_report(self, savefig=False):
        """
        Makes a labelled bar chart from a classification report that shows a F1-score for each class.

        Args:
        savefig: save classification report image to file (default=False).
        """
        class_report = classification_report(y_true=self.__y_true,
                                             y_pred=self.__y_preds,
                                             output_dict=True)
        class_f1_scores = {}

        for k, v in class_report.items():
            if k == "accuracy":
                break
            else:
                class_f1_scores[self.__class_names[int(k)]] = v["f1-score"]

        f1_scores = pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                                  "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 25))
        scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values)
        ax.set_yticks(range(len(f1_scores)))
        ax.set_yticklabels(f1_scores["class_names"])
        ax.set_xlabel("F1-score")
        ax.set_title(
            f"F1-scores for {len(self.__class_names)} different classes.")
        ax.invert_yaxis()

        for rect in scores:
            width = rect.get_width()
            ax.text(1.03*width, rect.get_y() + rect.get_height()/1.5,
                    f"{width:.4f}",
                    ha='center', va='bottom')
        if savefig:
            fig.savefig("confusion_matrix.png")

    def make_confusion_matrix(self, figsize=(10, 10), text_size=15, norm=False, savefig=False):
        """Makes a labelled confusion matrix comparing predictions and ground truth labels.

        If classes is passed, confusion matrix will be labelled, if not, integer class values
        will be used.

        Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).

        Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.

        Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
        y_pred=y_preds, # predicted labels
        classes=class_names, # array of class label names
        figsize=(15, 15),
        text_size=10)

        Note: The confusion matrix code is a remix of Scikit-Learn's
        plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
        """
        # Create the confustion matrix
        cm = confusion_matrix(self.__y_true, self.__y_preds)
        cm_norm = cm.astype("float") / \
            cm.sum(axis=1)[:, np.newaxis]  # normalize it
        # find the number of classes we're dealing with
        n_classes = cm.shape[0]

        # Plot the figure and make it pretty
        fig, ax = plt.subplots(figsize=figsize)
        # colors will represent how 'correct' a class is, darker == better
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Are there a list of classes?
        if self.__class_names:
            labels = self.__class_names
        else:
            labels = np.arange(cm.shape[0])

        # Label the axes
        ax.set(title="Confusion Matrix",
               xlabel="Predicted label",
               ylabel="True label",
               # create enough axis slots for each class
               xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               # axes will labeled with class names (if they exist) or ints
               xticklabels=labels,
               yticklabels=labels)

        # Make x-axis labels appear on bottom
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        plt.xticks(rotation=70, fontsize=text_size)
        plt.yticks(fontsize=text_size)

        # Set the threshold for different colors
        threshold = (cm.max() + cm.min()) / 2.

        # Plot the text on each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if norm:
                plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > threshold else "black",
                         size=text_size)
            else:
                plt.text(j, i, f"{cm[i, j]}",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > threshold else "black",
                         size=text_size)

        # Save the figure to the current working directory
        if savefig:
            fig.savefig("confusion_matrix.png")

    def get_preds_df(self, path=None):
        """Returns a Pandas DataFrame containing all predictions on test dataset"""
        #  Get all file paths from test dataset.

        if path == None:
            filepaths = np.zeros(len(self.__y_preds))
        else:
            filepaths = []
            for filepath in self.__test_data.list_files(path,
                                                        shuffle=False):
                filepaths.append(filepath.numpy())

        pred_df = pd.DataFrame({"img_path": filepaths,
                                "y_true": self.__y_true,
                                "y_preds": self.__y_preds,
                                "y_probs": self.__y_probs.max(axis=1),
                                "y_true_classnames": [self.__class_names[i] for i in self.__y_true],
                                "y_preds_classnames": [self.__class_names[i] for i in self.__y_preds],
                                "pred_correct": self.__y_true == self.__y_preds})
        return pred_df

    def plot_most_wrong_preds(self, numImgs=9, startIndex=0, figsize=(15, 10)):
        """
        From pred_df plots most wrong predictions.
        Args:
        numImgs: Int. Number of images to plot.
        startIndex: Int. Index number to start from DataFrame.
        figsize: Tuple. Plot size.
        """

        plt.figure(figsize=figsize)
        pred_df = self.get_preds_df()
        top_wrong = pred_df[pred_df['pred_correct'] ==
                            False].sort_values('y_probs', ascending=False)

        for i, row in enumerate(top_wrong[startIndex:startIndex+numImgs].itertuples()):
            plt.subplot(3, 3, i+1)
            img = self.load_and_prep_image(
                filename=row[1], img_size=self.__img_shape, scale=True)
            _, _, _, _, y_prob, y_true, y_pred, _ = row
            plt.imshow(img)
            plt.title(f"actual: {y_true}, pred: {y_pred} \nprob: {y_prob:.2f}")
            plt.axis(False)

    # Create callbacks.

    def create_tensorboard_callback(self, dir_name, experiment_name):
        """
        Creates a TensorBoard callback instand to store log files.

        Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"

        Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
        """
        log_dir = dir_name + "/" + experiment_name + "/" + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir
        )
        print(f"Saving TensorBoard log files to: {log_dir}")
        return tensorboard_callback

    def create_checkpoint_callback(self, dir_name, experiment_name):
        """
        Creates a checkpoint callback instance to store checkpoint files.

        Stores checkpoint files with the filepath:
        "dir_name/experiment_name/checkpoint.ckpt"

        Args:
        dir_name: target directory to store checkpoint files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
        """
        checkpoint_path = dir_name + "/" + experiment_name + "/checkpoint.ckpt"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                 save_weights_only=True,
                                                                 save_best_only=True,
                                                                 save_freq="epoch",
                                                                 monitor="val_loss",
                                                                 verbose=1)

        print(f"Saving checkpoint to: {checkpoint_path}")
        return checkpoint_callback

    def create_earlyStopping_callback(self, patience):
        earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience)

        return earlyStopping_callback
