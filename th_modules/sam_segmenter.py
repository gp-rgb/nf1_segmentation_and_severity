from matplotlib.image import imsave
import numpy as np  # linear algebra
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from matplotlib.backend_bases import MouseButton
from PIL import Image
import os

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

global input_coords, input_labels, input_boxes
#global x1, x2, y1, y2
#x1, x2, y1, y2 = None, None, None, None
input_coords = []
input_labels = []
input_boxes = []



# display functions for SAM
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def figure_mask_mask(skin,tumours,title=None,random_colour=False):
    fig = plt.figure()
    plt.imshow(skin,cmap='gray')
    ax=plt.gca()
    if tumours is not None:
        show_mask(tumours,ax,random_color=random_colour)
    if title is not None:
        ax.set_title(title)
    fig.show()

def figure_mask_image(image,mask=None,title=None,random_colour=False):
    fig = plt.figure()
    if len(image.shape) == 2: #single channel, b&w image
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    ax=plt.gca()
    if mask is not None:
        show_mask(mask,ax,random_color=random_colour)
    if title is not None:
        ax.set_title(title)
    fig.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(boxes, ax):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
        )


def mask_to_binary(mask):
    threshold = 0.5
    binary_mask = (mask >= threshold).astype(np.uint8)
    return binary_mask


def save_images_to_png(src_path, dest_path, image, masks, scores):
    original_image_name = src_path.split("/")[-1].split(".")[
        0
    ]  # filename with .jpeg extension removed
    newdir = f"{dest_path}/{original_image_name}/"
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    image_path = f"{newdir}/original.png"
    imsave(image_path, image)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        binary_mask = mask_to_binary(mask)
        mask_path = f"{newdir}/mask{i}_{score:.4f}.png"
        imsave(mask_path, binary_mask, cmap="gray")

    print("Images saved successfully.")


def init_sam_model(checkpoint_path="./sam_vith_model.pth") -> SamPredictor:
    # setting up the SAM model
    sam_checkpoint = "./sam_vith_model.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device=device)

    # linking the predictor object to the image
    predictor = SamPredictor(sam)

    # print("Finished Setting Up SAM Model")

    return predictor




def on_click(event):
    global input_coords, input_labels
    if event.inaxes:
        #input_coords = []
        #input_labels = []
        input_coords.append([event.xdata, event.ydata])
        if event.button is MouseButton.LEFT:
            # foreground
            input_labels.append(1)
        if event.button is MouseButton.RIGHT:
            # background
            input_labels.append(0)


def onselect(eclick, erelease):
    global input_coords, input_labels, input_boxes

    if eclick.button == 1 and erelease.button == 1:  # Left-click
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        #print(f"Box detected at {x1},{y1},{x2},{y2}")
        input_boxes = np.array([[round(x1), round(y1), round(x2), round(y2)]])
        # pop last item from the coords and labels list, 
        #   as specifying a box also adds a single positive point 
        #       (due to being mapped to a click!)
        input_coords.pop(-1) 
        input_labels.pop(-1)
        

    # print(f"Selected coordinates: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")


def get_masks_from_input(image, predictor: SamPredictor, multimask=True):
    global input_coords, input_labels, input_boxes
    input_coords = []
    input_labels = []
    input_boxes = []

    _, ax = plt.subplots()
    ax.imshow(image)
    binding_id = plt.connect("button_press_event", on_click)

    rs = RectangleSelector(ax, onselect, 
                           #drawtype="box", 
                           useblit=True,
                           minspanx=5, 
                           minspany=5)
    plt.show()
    #print(f"Coords: {input_coords}")
    #print(f"Labels: {input_labels}")
    print(f"Box: {input_boxes}")
    
    if len(input_boxes) != 0:
        box = np.array(input_boxes)
    else:
        box = None
    if len(input_coords) != 0:
        coords = np.array(input_coords)
        labels = np.array(input_labels)
    else:
        #print("input coord exception reached")
        coords = None
        labels = None

    print(f"Input Coordinates: {coords}")
    print(f"Input Labels: {labels}")
    print(f"Input Box: {box}")

    masks, scores, logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        box=box,
        multimask_output=multimask,
    )
    return masks, scores, logits, box, coords, labels


def show_sam_output(image, mask, score, coords, labels, box):
    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.imshow(image)
    show_mask(mask, ax)
    if box is not None:
        show_box(box, ax)
    if (coords is not None) and (labels is not None):
        show_points(coords, labels, ax)
    plt.title(f"Score: {score:.3f}", fontsize=18)
    plt.axis("off")
    plt.show()

def mask_to_mask(predictor:SamPredictor, mask):
    masks, _, _ = predictor.predict(
        mask_input=mask,
        multimask_output=False,
    )
    return masks