"""
Results depicted in various forms
"""

import random
import cv2
import torch
import matplotlib.pyplot as plt
from train import device, images, labels, imagesT, cnn

def label_text(label_int):
    """
    Return text representation of label_int (0, 1, 2, or 3)
    """
    if label_int == 0:
        return "reverse"
    if label_int == 1:
        return "forward"
    if label_int == 2:
        return "left"
    if label_int == 3:
        return "right"

    return "unrecognized"

def plot_predictions(model, imgsT, lbls):
    """
    Plot 12 random images with their predictions and labels. 9 correct, 3 incorrect.
    Args:
        model: torch NN model
        imgsT: images tensor
        lbls: labels array
    """
    plt.figure(figsize=(13,14))

    correct = 0
    incorrect = 0

    while correct < 9 or incorrect < 3:
        idx = random.randint(0, 3861)
        img = imgsT[idx].unsqueeze(0).unsqueeze(0)
        pred = int(model(img).max(1, keepdim=True)[1])
        lbl = int(lbls[idx])

        if pred == lbl and correct < 9:
            correct += 1
            plt.subplot(4, 3, correct)
            plt.imshow(imgsT[idx].cpu().numpy(), cmap='gray')
            plt.title("Prediction: {} | Actual: {}".format(label_text(pred), label_text(lbl)))
        elif pred != lbl and incorrect < 3:
            incorrect += 1
            plt.subplot(4, 3, incorrect + 9)
            plt.imshow(imgsT[idx].cpu().numpy(), cmap='gray')
            plt.title("Prediction: {} | Actual: {}".format(label_text(pred), label_text(lbl)))

    plt.subplots_adjust(hspace=0.3, wspace=0)
    plt.show()

def constr_vid_custom(model):
    """
    Create labelled images on custom data with model's prediction and
    reconstruct video out of images
    """
    # create labelled images
    for i in range(317):
        img = cv2.imread("data/custom/frame%d.jpg" %i, 0)

        # make prediction on normalized image
        imgT = torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0)
        imgT = imgT - (imgT.max() + imgT.min()) / 2
        imgT = imgT / imgT.max()
        pred = label_text(int(model(imgT).max(1, keepdim=True)[1]))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, pred, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite("data/custom-labelled/Rframe%d.jpg" %i, img)

    # convert labelled images to video

    img = cv2.imread('data/custom-labelled/Rframe0.jpg', -1)
    height, width = img.shape
    size = (width, height)
    FPS = 15

    out = cv2.VideoWriter('result-custom.avi', cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    for i in range(317):
        img = cv2.imread('data/custom-labelled/Rframe%d.jpg' %i)
        out.write(img)

    out.release()

def constr_vid_orig(model, imgs):
    """
    Create labelled images on original data with model's prediction and
    reconstruct video out of images
    """
    cv_ims = imgs - imgs.min()

    # create labelled images

    for i in range(3862):
        img = cv_ims[i]

        # make prediction on normalized image
        imgT = torch.from_numpy(img).to(device).float().unsqueeze(0).unsqueeze(0)
        imgT = imgT - (imgT.max() + imgT.min()) / 2
        imgT = imgT / imgT.max()
        pred = label_text(int(model(imgT).max(1, keepdim=True)[1]))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, pred, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite("data/orig-labelled/Rframe%d.jpg" %i, img)

    # convert labelled images to video

    img = cv2.imread('data/orig-labelled/Rframe0.jpg', -1)
    height, width = img.shape
    size = (width, height)
    FPS = 15

    out = cv2.VideoWriter('result-orig.avi', cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

    for i in range(3862):
        img = cv2.imread('data/orig-labelled/Rframe%d.jpg' %i)
        out.write(img)

    out.release()

if __name__ == '__main__':
    plot_predictions(cnn, imagesT, labels)
    constr_vid_custom(cnn)
    constr_vid_orig(cnn, images)
