import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import random 
import os
import datetime
import zipfile
import tensorflow as tf


def plot_decision_boundary(model, X, y):
    '''
    Plot the decision boundary created by a model predicting on X
    Args:
        model: trained model
        input: X
        output: y
    '''
    # Define the axis boundaries of the plot
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # create a meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    # create X value (to make predictions on these values)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predicts
    y_pred = model.predict(x_in)

    # check for multi-class
    if len(y_pred[0]) > 1:
        print('doing multi-class classification')
        # reshape the predictions.
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:  # binary classification
        print('doing binary classification')
        y_pred = np.round(y_pred).reshape(xx.shape)

    # plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_pretty_confusion_matrix(y_test, y_preds, classes=None, 
                            figsize=(10,10), text_size=20,
                            show_prob=False, save_fig=True):
    """
    Plots a beatiful confusion matrix 

    Args:
        y_test: ground_truth labels
        y_preds: predicted labels
        classes: class_names
        fig_size: (10,10) default
        text_size: font size
    Returns:
        None:
    """
    # create confusion matrix
    cm = confusion_matrix(y_test, tf.round(y_preds))
    cm_norm = cm.astype('float') / cm.sum(axis=1)
    n_classes = cm.shape[0]

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    # create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # create classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # label the axes
    ax.set(title="Confusion Matrix", 
        xlabel='Predicted Label',
        ylabel='True Label',
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

    # set X-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ### change (plot xlabels vertically)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # plot text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_prob:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i,j] > threshold else "black",
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i,j] > threshold else "black",
                size=text_size)

    if save_fig:
        plt.savefig("confusion_matrix.png")
            

def plot_random_images(model, images, true_labels, classes, rows=1, cols=1, scale=2):
  """
  picks a random image, plots it and labels it with prediction & true labels
  """
  assert rows >=1 and cols >=1, "rows and cols must be non-zero positive integers"
  # total number of images to be plotted
  n = rows * cols
  fig = plt.figure(figsize=(rows*scale, cols*scale), tight_layout=True)
  for i in range(n):
    ax = plt.subplot(rows, cols, i+1)
    index = random.randint(0, len(images))

    # target image
    target_image = images[index]

    # Prediction
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[index]]

    # plot the image
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Text colors
    if pred_label == true_label:
      color = "green"
    else:
      color = "red"

    plt.title("Pred: {} {:2.0f}% \n (True: {})".format(pred_label,
                                                    100 * tf.reduce_max(pred_probs),
                                                    true_label), color=color)
    plt.axis('off')


def plot_training_curves(history):
    """
    plots separate loss and accuracy curves
    Args:
        history: history object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training')
    plt.plot(epochs, val_loss, label='validation')
    plt.title('Losses')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    plt.figure()
    plt.plot(epochs, accuracy, label='training')
    plt.plot(epochs, val_accuracy, label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')


def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    loads and preprocess image - resize & normalize
    Args:
        filename (str): path to target image
        img_shape (int): image size
        scale (bool): whether to scale the image
    """
    # read the file
    img = tf.io.read_file(filename)

    # decode the file into an image
    img = tf.image.decode_image(img, channels=3)

    # resize the image
    img = tf.image.resize(img, size=(img_shape, img_shape))

    # scale the image
    if scale:
        img = img / 255.
    return img


def pred_and_plot(model, filename, class_names):
    """
    loads an image and predicts its class label using the model
    """
    # import the image
    img = load_and_prep_image(filename)

    # make prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:        # multi-class classification
        pred_class = class_names[tf.argmax(pred[0])]
        pred_prob = tf.math.reduce_max(pred, axis=1)
    else:       # binary class classification
        pred_class = class_names[int(tf.round(pred))]
        pred_prob = pred[0]

    # plot the image & shows it predicted class label
    plt.imshow(img)
    plt.title(f"Predicted class: {pred_class}")
    plt.axis(False)
    return pred_prob


def walk_through_folder(folder_path):
    """
    Walks through a given folder 
    Args:
        folder_path: 
    """
    for dirpath, dirnames, filenames in os.walk(folder_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def create_tensorboard_callback(dir_name, experiment_name):
    """
    create tensorboard log folders
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_callback 


def unzip_data(filename: str) -> None:
    """
    unzip the the zip folder
    """
    # unzip our data
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall()
    zip_ref.close()


def compare_histories(original_history, new_history, initial_epochs=5):
  """
  Comparing two tensorflow histories
  """
  acc = original_history.history['accuracy']
  loss = original_history.history['loss']
  val_acc = original_history.history['val_accuracy']
  val_loss = original_history.history['val_loss']

  # combine original history metrics with new_history metrics
  total_acc = acc + new_history.history['accuracy']
  total_loss = loss + new_history.history['loss']
  total_val_acc = val_acc + new_history.history['val_accuracy']
  total_val_loss = val_loss + new_history.history['val_loss']

  # make plots
  plt.figure(figsize=(8,8))
  plt.subplot(2,1,1)
  plt.plot(total_acc, label='training_accuracy')
  plt.plot(total_val_acc, label='validation_accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='lower right')
  plt.title('Training & Validation Accuracy')

  plt.figure(figsize=(8,8))
  plt.subplot(2,1,2)
  plt.plot(total_loss, label='training_accuracy')
  plt.plot(total_val_loss, label='validation_accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
  plt.legend(loc='upper right')
  plt.xlabel('Epochs')
  plt.title('Training & Validation Loss')
  
