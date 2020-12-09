"""
Results depicted in various forms
"""

import cv2
import torch
import matplotlib.pyplot as plt
from train import device, images, cnn

def label_text(label_int):
    """
    Return text representation of label_int
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

def plot_predictions(model, imgs):
    """
    Plot 3 predictions for fw, right, left
    """
    f_idx = [0, 1054, 3840]
    l_idx = [214, 2449, 3691]
    r_idx = [41, 1220, 2718]

    plt.figure(figsize=(10,10))

    for i in range(9):
        if i < 3:
            pred = label_text(int(model(imgs[f_idx[i]].unsqueeze(0).unsqueeze(0)).max(1, keepdim=True)[1]))
            plt.subplot(3, 3, i+1)
            plt.imshow(imgs[f_idx[i]].cpu().numpy(), cmap='gray')
            plt.title("Prediction: {}".format(pred))
        elif i < 6:
            pred = label_text(int(model(images[l_idx[i-3]].unsqueeze(0).unsqueeze(0)).max(1, keepdim=True)[1]))
            plt.subplot(3, 3, i+1)
            plt.imshow(imgs[l_idx[i-3]].cpu().numpy(), cmap='gray')
            plt.title("Prediction: {}".format(pred))
        else:
            pred = label_text(int(model(imgs[r_idx[i-6]].unsqueeze(0).unsqueeze(0)).max(1, keepdim=True)[1]))
            plt.subplot(3, 3, i+1)
            plt.imshow(imgs[r_idx[i-6]].cpu().numpy(), cmap='gray')
            plt.title("Prediction: {}".format(pred))

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
    plot_predictions(cnn, images)
    constr_vid_custom(cnn)
    constr_vid_orig(cnn, images)
