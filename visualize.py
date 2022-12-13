"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import string
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

from matplotlib.colors import colorConverter
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


from utils import mrcnn_utils as utils

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns


import os
import logging
import os
import random
from itertools import cycle

import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix

random.seed(9001)
import numpy as np

import importlib
import matplotlib.pyplot as plt
import matplotlib

from utils import config_utils

from copy import deepcopy
import pprint

from skimage.measure import find_contours
from matplotlib.patches import Polygon

############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None, interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        # plt.imshow(image.astype(np.uint8), cmap=cmap, norm=norm, interpolation=interpolation)
        plt.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, savepath=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]

    if not N:
        print("*** No instances to display ***")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    # masked_image = image.astype(np.uint32).copy()
    masked_image = image.copy() # for nifti

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        # if show_mask:
        #     masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image[:,:,0][...,np.newaxis], cmap='gray') # for nifti

    # show_mask functionality modified for nifti
    if show_mask:
        # generate the colors for your colormap
        color1 = colorConverter.to_rgba('black')
        color2 = colorConverter.to_rgba('red')
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

        cmap2._init() # create the _lut array, with rgba values
        ax.imshow(np.squeeze(masks), cmap=cmap2, alpha = 0.1)

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath)

    if auto_show:
        plt.show()

    plt.close()
    

def display_instances_all_mods(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(18,6), ax=None,
                      show_mask=False, show_bbox=True,
                      savepath=None, suptitle = None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]

    if not N:
        print("*** No instances to display ***")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False

    if not ax:
        fig = plt.figure(figsize=figsize)
        gs1 = gridspec.GridSpec(1, image.shape[0])
        gs1.update(wspace=0)  # set the spacing between axes.
        axes = [plt.subplot(gs1[i]) for i in range(image.shape[0])]

        # fig, axes = plt.subplots(1, image.shape[-1], figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[1:]

    for ax in axes:
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

        masked_image = image.copy() # for nifti

        for i in range(N):
            color = colors[i]

            # Bounding box

            # Skip this instance. Has no bbox. Likely lost in image cropping.
            if not np.any(boxes[i]): continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="dashed", edgecolor=color, facecolor='none')
                ax.add_patch(p)

            # Label
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label

            ax.text(x1, y1 - 5, caption, color='w', size=15, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            # if show_mask:
            #     masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    for idx, i in enumerate(axes):
        axes[idx].imshow(masked_image[idx, :, :], cmap='gray')  # for nifti

    # generate the colors for your colormap
    color1 = colorConverter.to_rgba('black')
    color2 = colorConverter.to_rgba('red')
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)

    cmap2._init()  # create the _lut array, with rgba values

    for ax in axes:
        if show_mask: ax.imshow(np.squeeze(masks), cmap=cmap2, alpha=0.1)

    # print("suptitle", suptitle)
    if suptitle: fig.suptitle(str("\n".join(suptitle)), fontsize=15)
    if savepath: plt.savefig(savepath,  bbox_inches='tight')

    plt.close()

def display_instances_all_mods_only_bbox(image, boxes, class_ids, class_names, gt_mask,
                              scores=None, title="", show_bbox=True, figsize=(18,6), ax = None,
                              savepath=None, suptitle = None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]

    if not N:
        print("*** No instances to display ***")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False

    if not ax:
        fig = plt.figure(figsize=figsize)
        gs1 = gridspec.GridSpec(1, image.shape[0])
        gs1.update(wspace=0)  # set the spacing between axes.
        axes = [plt.subplot(gs1[i]) for i in range(image.shape[0])]

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[1:]

    for ax in axes:
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

        masked_image = image.copy() # for nifti

        for i in range(N):
            color = colors[i]

            # Bounding box

            # Skip this instance. Has no bbox. Likely lost in image cropping.
            if not np.any(boxes[i]): continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="dashed", edgecolor=color, facecolor='none')
                ax.add_patch(p)

            # Label
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label

            ax.text(x1, y1 - 5, caption, color='w', size=15, backgroundcolor="none")

        # # GT Mask
        # mask = gt_mask # 128 x 128
        
        # # Mask Polygon
        # # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor='g')
        #     ax.add_patch(p)

    for idx, i in enumerate(axes):
        axes[idx].imshow(masked_image[idx, :, :], cmap='gray')  # for nifti

    if suptitle: fig.suptitle(str("\n".join(suptitle)), fontsize=15)
    if savepath: plt.savefig(savepath,  bbox_inches='tight')

    plt.close()




def add_polygon_mask_outline_to_ax(ax, mask, color):
    '''
    Takes 2d (128 x 128) mask as input and adds it to ax as patch
    '''
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)
    return ax


def plot_3d_prediction_from_results_dict(batch, results_dict, cf, suptitle = None, outfile= None):
    """
    plot the input images, ground truth annotations, and output predictions of a batch. If 3D batch, plots a 2D projection
    of one randomly sampled element (patient) in the batch. Since plotting all slices of patient volume blows up costs of
    time and space, only a section containing a randomly sampled ground truth annotation is plotted.
    :param batch: dict with keys: 'data' (input image), 'seg' (pixelwise annotations), 'pid'
    :param results_dict: list over batch element. Each element is a list of boxes (prediction and ground truth),
    where every box is a dictionary containing box_coords, box_score and box_type.
    """
    if outfile is None:
        outfile = os.path.join(cf.plot_dir, 'pred_example_{}.png'.format(cf.fold))

    data = batch['data'] # (1, 3, 128, 128, 128)
    segs = batch['seg'] # (1, 1, 128, 128, 128), unique elements = {0: x, 1: y}
    pids = batch['pid'] # 

    # for 3D, repeat pid over batch elements.
    if len(set(pids)) == 1:
        pids = [pids] * data.shape[0]

    seg_preds = results_dict['seg_preds'] # (1, 1, 128, 128, 128), unique elements = {0: x, 1: y}

    # if test_aug = True in config, then seg_pred might have multiple predicted segmentations for single patient (i.e. shape[1] > 1)
    # in that case, take mean of all the segmentation masks (i.e. take mean across axis = 1)
    if seg_preds.shape[1] > 1:
        seg_preds = np.mean(seg_preds, 1)[:, None]
    
    roi_results = deepcopy(results_dict['boxes'])

    # keep only top-1 box
    highest_pred_confidence = np.max(np.array([i['box_score'] for i in roi_results[0] if i['box_type'] == 'det']))
    roi_results = [[i for i in roi_results[0] if (i['box_type'] == 'det' and i['box_score'] == highest_pred_confidence) or (i['box_type'] == 'gt')]]
    
    if cf.dim == 3:
        # Randomly sampled one patient of batch and project data into 2D slices for plotting.
        patient_ix = 0
        data = np.transpose(data[patient_ix], axes=(3, 0, 1, 2))

        # select interesting foreground section to plot.
        gt_boxes = [box['box_coords'] for box in roi_results[patient_ix] if box['box_type'] == 'gt']
        if len(gt_boxes) > 0:
            z_cuts = [int(gt_boxes[0][4]), int(gt_boxes[0][5])]
        else:
            z_cuts = [data.shape[0]//2 - 5, int(data.shape[0]//2 + np.min([10, data.shape[0]//2]))]
        
        p_roi_results = roi_results[patient_ix]
        roi_results = [[] for _ in range(data.shape[0])]

        # iterate over cubes and spread across slices.
        for box in p_roi_results:
            b = box['box_coords']
            # dismiss negative anchor slices.
            slices = np.round(np.unique(np.clip(np.arange(b[4], b[5] + 1), 0, data.shape[0]-1)))
            for s in slices:
                roi_results[int(s)].append(box)
                roi_results[int(s)][-1]['box_coords'] = b[:4]

        roi_results = roi_results[z_cuts[0]: z_cuts[1]]
        # print("3 --> ")
        # pprint.pprint(roi_results)

        data = data[z_cuts[0]: z_cuts[1]]
        segs = np.transpose(segs[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]
        seg_preds = np.transpose(seg_preds[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]
        pids = [pids[patient_ix]] * data.shape[0]

    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 2, 3]:
            assert data.shape[i] == segs.shape[i] == seg_preds.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement!'
                      'Shapes {} vs. {} vs {}'.format(data.shape, segs.shape, seg_preds.shape))


    # show_arrays = np.concatenate([data, segs, seg_preds, data[:, 0][:, None]], axis=1).astype(float)
    show_arrays = np.concatenate([data], axis=1).astype(float)
    # print(data.shape)
    # print(segs.shape)
    # print(seg_preds.shape)
    # print(show_arrays.shape)

    step_size = 10 # every 'step_size'th column will be plotted
    approx_figshape = ((4/step_size) * show_arrays.shape[0], 4 * show_arrays.shape[1]+2)
    # print("approx_figshape", approx_figshape)
    fig = plt.figure(figsize=approx_figshape)
    
    gs = gridspec.GridSpec(show_arrays.shape[1] + 1, len(range(0, show_arrays.shape[0], step_size)))
    gs.update(wspace=0.01, hspace=0.01)

    for bidx, b in enumerate(range(0, show_arrays.shape[0], step_size)): # columns
        for m in range(show_arrays.shape[1]): # rows

            ax = plt.subplot(gs[m, bidx])
            
            # This code snippet is to hide axes but show row-labels ## >>>>>
            ax.xaxis.set_visible(False)            
            ax.get_yaxis().set_ticks([])
            if bidx == 0:
                ax.set_ylabel(cf.config["training_modalities"][m].split('_')[0], rotation=90, fontsize=20)
            ########################################################## >>>>>

            if m < show_arrays.shape[1]: 
                arr = show_arrays[b, m]
                seg = np.squeeze(segs[b])
                seg_pred = np.squeeze(seg_preds[b])

            # rows for data modalities
            if m < data.shape[1]: 
                plt.imshow(arr, cmap='gray', vmin=None, vmax=None)   
                # add polygonal mask outline to ax
                add_polygon_mask_outline_to_ax(ax, seg, color = 'b') 
                add_polygon_mask_outline_to_ax(ax, seg_pred, color = 'r') 


            # rows for gt and predicted seg
            else: 
                plt.imshow(arr, cmap='viridis', vmin=0, vmax=cf.num_seg_classes - 1)            
            
            # # ax titles
            # if m == 0: plt.title('{}'.format(pids[b][:10]), fontsize=20)

            ############## Bboxes ##############
            # if m >= (data.shape[1]):
            for box in roi_results[b]:
                if (box['box_type'] != 'patient_tn_box') and (box['box_type'] != 'gt'): # don't plot true negative dummy boxes AND gt boxes.
                    coords = box['box_coords']
                    if box['box_type'] == 'det':
                        # dont plot background preds or low confidence boxes.
                        if box['box_pred_class_id'] > 0 and box['box_score'] > 0.1:
                            plot_text = True
                            score = np.max(box['box_score'])
                            labels_to_use = ['Mutant', 'WT']
                            score_text = '{} {:.3f}'.format(labels_to_use[box['box_pred_class_id']-1], score)
                            
                            score_font_size = 15
                            text_color = 'w'
                            text_x = coords[1] + 10*(box['box_pred_class_id'] -1) #avoid overlap of scores in plot.
                            text_y = coords[2] + 9
                        else:
                            continue
                    # elif box['box_type'] == 'gt':
                    #     plot_text = True
                    #     score_text = int(box['box_label'])
                    #     score_font_size = 7
                    #     text_color = 'r'
                    #     text_x = coords[1]
                    #     text_y = coords[0] - 1
                    else:
                        plot_text = False

                    color = cf.box_color_palette[box['box_type']]
                    plt.plot([coords[1], coords[3]], [coords[0], coords[0]], color=color, linewidth=2, linestyle="dashed") # up
                    plt.plot([coords[1], coords[3]], [coords[2], coords[2]], color=color, linewidth=2, linestyle="dashed") # down
                    plt.plot([coords[1], coords[1]], [coords[0], coords[2]], color=color, linewidth=2, linestyle="dashed") # left
                    plt.plot([coords[3], coords[3]], [coords[0], coords[2]], color=color, linewidth=2, linestyle="dashed") # right
                    
                    if plot_text: plt.text(text_x, text_y, score_text, fontsize=score_font_size, color=text_color)

            ############################
    
    if suptitle: fig.suptitle(str("\n".join(suptitle)), fontsize=20)
    print("Saving at: ", outfile)
    plt.savefig(outfile, bbox_inches = 'tight')   
    plt.close(fig)


def plot_3d_prediction_from_results_dict_without_predmask(batch, results_dict, cf, suptitle = None, outfile= None):
    """
    plot the input images, ground truth annotations, and output predictions of a batch. If 3D batch, plots a 2D projection
    of one randomly sampled element (patient) in the batch. Since plotting all slices of patient volume blows up costs of
    time and space, only a section containing a randomly sampled ground truth annotation is plotted.
    :param batch: dict with keys: 'data' (input image), 'seg' (pixelwise annotations), 'pid'
    :param results_dict: list over batch element. Each element is a list of boxes (prediction and ground truth),
    where every box is a dictionary containing box_coords, box_score and box_type.
    """
    if outfile is None:
        outfile = os.path.join(cf.plot_dir, 'pred_example_{}.png'.format(cf.fold))

    data = batch['data'] # (1, 3, 128, 128, 128)
    segs = batch['seg'] # (1, 1, 128, 128, 128), unique elements = {0: x, 1: y}
    pids = batch['pid'] # 

    # for 3D, repeat pid over batch elements.
    if len(set(pids)) == 1:
        pids = [pids] * data.shape[0]
    
    roi_results = deepcopy(results_dict['boxes'])

    # keep only top-1 box
    highest_pred_confidence = np.max(np.array([i['box_score'] for i in roi_results[0] if i['box_type'] == 'det']))
    roi_results = [[i for i in roi_results[0] if (i['box_type'] == 'det' and i['box_score'] == highest_pred_confidence) or (i['box_type'] == 'gt')]]
    
    # Randomly sampled one patient of batch and project data into 2D slices for plotting.
    patient_ix = 0
    data = np.transpose(data[patient_ix], axes=(3, 0, 1, 2))

    # select interesting foreground section to plot.
    gt_boxes = [box['box_coords'] for box in roi_results[patient_ix] if box['box_type'] == 'gt']
    if len(gt_boxes) > 0:
        z_cuts = [int(gt_boxes[0][4]), int(gt_boxes[0][5])]
    else:
        z_cuts = [data.shape[0]//2 - 5, int(data.shape[0]//2 + np.min([10, data.shape[0]//2]))]
    
    p_roi_results = roi_results[patient_ix]
    roi_results = [[] for _ in range(data.shape[0])]

    # iterate over cubes and spread across slices.
    for box in p_roi_results:
        b = box['box_coords']
        # dismiss negative anchor slices.
        slices = np.round(np.unique(np.clip(np.arange(b[4], b[5] + 1), 0, data.shape[0]-1)))
        for s in slices:
            roi_results[int(s)].append(box)
            roi_results[int(s)][-1]['box_coords'] = b[:4]

    roi_results = roi_results[z_cuts[0]: z_cuts[1]]
    # print("3 --> ")
    # pprint.pprint(roi_results)

    data = data[z_cuts[0]: z_cuts[1]]
    segs = np.transpose(segs[patient_ix], axes=(3, 0, 1, 2))[z_cuts[0]: z_cuts[1]]
    pids = [pids[patient_ix]] * data.shape[0]

    try:
        # all dimensions except for the 'channel-dimension' are required to match
        for i in [0, 2, 3]:
            assert data.shape[i] == segs.shape[i] == seg_preds.shape[i]
    except:
        raise Warning('Shapes of arrays to plot not in agreement!'
                      'Shapes {} vs. {} vs {}'.format(data.shape, segs.shape, seg_preds.shape))


    # show_arrays = np.concatenate([data, segs, seg_preds, data[:, 0][:, None]], axis=1).astype(float)
    show_arrays = np.concatenate([data], axis=1).astype(float)
    # print(data.shape)
    # print(segs.shape)
    # print(show_arrays.shape)

    step_size = 10 # every 'step_size'th column will be plotted
    approx_figshape = ((4/step_size) * show_arrays.shape[0], 4 * show_arrays.shape[1]+2)
    # print("approx_figshape", approx_figshape)
    fig = plt.figure(figsize=approx_figshape)
    
    gs = gridspec.GridSpec(show_arrays.shape[1] + 1, len(range(0, show_arrays.shape[0], step_size)))
    gs.update(wspace=0.01, hspace=0.01)

    for bidx, b in enumerate(range(0, show_arrays.shape[0], step_size)): # columns
        for m in range(show_arrays.shape[1]): # rows

            ax = plt.subplot(gs[m, bidx])
            
            # This code snippet is to hide axes but show row-labels ## >>>>>
            ax.xaxis.set_visible(False)            
            ax.get_yaxis().set_ticks([])
            if bidx == 0:
                ax.set_ylabel(cf.config["training_modalities"][m].split('_')[0], rotation=90, fontsize=20)
            ########################################################## >>>>>

            if m < show_arrays.shape[1]: 
                arr = show_arrays[b, m]
                seg = np.squeeze(segs[b])
                

            # rows for data modalities
            if m < data.shape[1]: 
                plt.imshow(arr, cmap='gray', vmin=None, vmax=None)   
                # add polygonal mask outline to ax
                add_polygon_mask_outline_to_ax(ax, seg, color = 'b') 
                


            # rows for gt and predicted seg
            else: 
                plt.imshow(arr, cmap='viridis', vmin=0, vmax=cf.num_seg_classes - 1)            
            
            # # ax titles
            # if m == 0: plt.title('{}'.format(pids[b][:10]), fontsize=20)

            ############## Bboxes ##############
            # if m >= (data.shape[1]):
            for box in roi_results[b]:
                if (box['box_type'] != 'patient_tn_box') and (box['box_type'] != 'gt'): # don't plot true negative dummy boxes AND gt boxes.
                    coords = box['box_coords']
                    if box['box_type'] == 'det':
                        # dont plot background preds or low confidence boxes.
                        if box['box_pred_class_id'] > 0 and box['box_score'] > 0.1:
                            plot_text = True
                            score = np.max(box['box_score'])
                            labels_to_use = ['Mutant', 'WT']
                            score_text = '{} {:.3f}'.format(labels_to_use[box['box_pred_class_id']-1], score)
                            
                            score_font_size = 15
                            text_color = 'w'
                            text_x = coords[1] + 10*(box['box_pred_class_id'] -1) #avoid overlap of scores in plot.
                            text_y = coords[2] + 9
                        else:
                            continue
                    # elif box['box_type'] == 'gt':
                    #     plot_text = True
                    #     score_text = int(box['box_label'])
                    #     score_font_size = 7
                    #     text_color = 'r'
                    #     text_x = coords[1]
                    #     text_y = coords[0] - 1
                    else:
                        plot_text = False

                    color = cf.box_color_palette[box['box_type']]
                    plt.plot([coords[1], coords[3]], [coords[0], coords[0]], color=color, linewidth=2, linestyle="dashed") # up
                    plt.plot([coords[1], coords[3]], [coords[2], coords[2]], color=color, linewidth=2, linestyle="dashed") # down
                    plt.plot([coords[1], coords[1]], [coords[0], coords[2]], color=color, linewidth=2, linestyle="dashed") # left
                    plt.plot([coords[3], coords[3]], [coords[0], coords[2]], color=color, linewidth=2, linestyle="dashed") # right
                    
                    if plot_text: plt.text(text_x, text_y, score_text, fontsize=score_font_size, color=text_color)

            ############################
    
    if suptitle: fig.suptitle(str("\n".join(suptitle)), fontsize=20)
    plt.savefig(outfile, bbox_inches = 'tight')   
    plt.close(fig)

# def display_instances(image, boxes, masks, class_ids, class_names,
#                                   scores=None, title="",
#                                   figsize=(16, 16), ax=None,
#                                   show_mask=False, show_bbox=True,
#                                   colors=None, captions=None, savepath=None):
#     """
#     modified display_instances for nifti files
#     """
#
#     image = image[:,:,0][...,np.newaxis]
#     # print("display_instances_for_nifti ==", image.shape)
#     # print("display_instances_for_nifti masks ==", masks.shape)
#
#     # Number of instances
#     N = boxes.shape[0]
#
#     if not N:
#         print("*** No instances to display ***")
#     else:
#         assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
#
#     auto_show = False
#
#     if not ax:
#         fig = plt.figure(figsize = figsize)
#         gs1 = gridspec.GridSpec(1,2)
#         gs1.update(wspace=0.025) # set the spacing between axes.
#
#         ax0 =  plt.subplot(gs1[0])
#         ax1 =  plt.subplot(gs1[1])
#
#         auto_show = True
#
#     # Generate random colors
#     colors = colors or random_colors(N)
#
#     height, width = image.shape[:2]
#     ax0.axis('off')
#     ax0.set_title(title)
#
#     for i in range(N):
#         color = colors[i]
#
#         # Bounding box
#         if not np.any(boxes[i]):
#             # Skip this instance. Has no bbox. Likely lost in image cropping.
#             continue
#         x1, y1, x2, y2 = boxes[i]
#         height =  -(x1-y1)
#         width = -(y2-x2)
#         # print(height, width)
#
#         if show_bbox:
#             # a = [[], [x2, y2]]
#             # plt.scatter(x1,y1, marker='o', color='r')
#             plt.scatter(x1,y2, marker='o', color='r') # top left of bbox
#             # plt.scatter(x2,y1, marker='o', color='b')
#             # plt.scatter(x2,y2, marker='o', color='m')
#             #
#             # plt.scatter(y1, x1, marker='o', color='r')
#             plt.scatter(y1, x2, marker='o', color='g') # bottom right of bbox
#             # plt.scatter(y2, x1, marker='o', color='b')
#             # plt.scatter(y2, x2, marker='o', color='m')
#
#             # ax0.add_patch(patches.Rectangle((y1, x2),width, height,linewidth=2,edgecolor='r',facecolor='none', ls = '--'))
#             ax0.add_patch(patches.Rectangle((x1, y2), height, width, linewidth=2, edgecolor='r', facecolor='none', ls='--'))
#             # ax1.add_patch(patches.Rectangle((x1, y2), height, width, linewidth=2, edgecolor='m', facecolor='none', ls='--'))
#
#         # Label
#         if not captions:
#             class_id = class_ids[i]
#             score = scores[i] if scores is not None else None
#             label = class_names[class_id]
#             caption = "{} {:.3f}".format(label, score) if score else label
#         else:
#             caption = captions[i]
#         ax0.text(x1, y2 + 8, caption, color='w', size=15, backgroundcolor="none")
#
#     # overlay mask
#     # generate the colors for your colormap
#     color1 = colorConverter.to_rgba('black')
#     color2 = colorConverter.to_rgba('red')
#     cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
#
#     cmap2._init() # create the _lut array, with rgba values
#
#     ax0.imshow(np.squeeze(image).T, cmap='gray', origin="lower")
#     # ax0.imshow(np.squeeze(masks).T, cmap=cmap2, origin="lower", alpha = 0.3)
#     ax1.imshow(np.squeeze(masks).T, cmap='gray', origin="lower")
#     ax1.axis('off')
#
#     if savepath: plt.savefig(savepath)
#
#     if auto_show: plt.show()
#
#     plt.close()
    

def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    # ax.imshow(masked_image)
    ax.imshow(masked_image[:, :, 0][..., np.newaxis], cmap='gray')  # for nifti

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


# def plot_precision_recall(AP, precisions, recalls):
#     """Draw the precision-recall curve.

#     AP: Average precision at IoU >= 0.5
#     precisions: list of precision values
#     recalls: list of recall values
#     """
#     # Plot the Precision-Recall curve
#     _, ax = plt.subplots(1)
#     ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
#     ax.set_ylim(0, 1.1)
#     ax.set_xlim(0, 1.1)
#     _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    # masked_image = image.astype(np.uint32).copy()
    masked_image = image.copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    # ax.imshow(masked_image.astype(np.uint8))
    ax.imshow(masked_image[:,:,0][...,np.newaxis], cmap='gray') # for nifti


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    print(table)
    # display_table(table)


def plot_cm_from_df(df, config, ax = None, savepath = None):

    cohort_suffix = df["cohort"].unique().tolist()
    if len(cohort_suffix) > 1:
        raise ValueError("There are more than one type of cohort present in excel file")

    if cohort_suffix[0] == 'val':
            cmap = 'Blues'
    elif cohort_suffix[0] == 'test':
        cmap = 'Reds'
    else:
        cmap = 'Greens'

    y_true = df[config['marker_column']].tolist()
    y_pred = df['Prediction'].tolist()

    # List of labels which are present in either true or predicted classes
    unique_true_labels = df[config['marker_column']].unique().tolist()
    unique_pred_labels = df['Prediction'].unique().tolist()

    # Need to add ['BG'] because in some cases, there are **no instances to display** and only background is predicted
    labels_to_use = [i for i in config['labels_to_use'] if i in list(set(unique_true_labels + unique_pred_labels))] + ['BG']

    # List of labels which are present in only true classes
    classes_present_true = [tuple([idx,i]) for idx, i in enumerate(labels_to_use) if i in list(set(unique_true_labels))]

    # List of labels which are present in only pred classes
    classes_present_pred = [tuple([idx,i]) for idx, i in enumerate(labels_to_use) if i in list(set(unique_pred_labels))]


    cm = confusion_matrix(y_true, y_pred, labels=labels_to_use)

    # Remove all zero rows from cm (i.e. rows of classes that are not present in true class)
    # https://stackoverflow.com/questions/11188364/remove-zero-lines-2-d-numpy-array
    cm = cm[~np.all(cm == 0, axis=1)] 

    # Remove all zero columns from cm (i.e. columns of classes that have not been predicted at all)
    # https://stackoverflow.com/questions/51769962/find-and-delete-all-zero-columns-from-numpy-array-using-fancy-indexing
    cm = np.delete(cm, np.argwhere(np.all(cm[..., :] == 0, axis=0)), axis=1)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            s = cm_sum[i]

            if c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

    cm_perc = pd.DataFrame(cm_perc, index=classes_present_true, columns=classes_present_pred)
    cm_perc.index.name = 'True label'
    cm_perc.columns.name = 'Predicted label'

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.set_title('Confusion matrix [{}]'.format(cohort_suffix), fontsize='25')
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    hmap = sns.heatmap(cm_perc, cmap=cmap, annot=annot, square=True, fmt='', ax=ax, annot_kws={"size": 30}, cbar_kws={"shrink": 0.75}, linewidths=0.1, vmax=100, vmin =0, linecolor='gray')

    hmap.set_xticklabels([i[1] for i in classes_present_pred], fontsize=20)
    hmap.set_yticklabels([i[1] for i in classes_present_true], fontsize=20)

    # use matplotlib.colorbar.Colorbar object
    cbar = hmap.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    for _, spine in hmap.spines.items():
        spine.set_visible(True)

    if savepath is None: 
        return ax
    else:
        # Save plot
        plt.suptitle('{}'.format("/".join(savepath.split('/')[4:])), fontsize=15, y=0.9)
        # plt.tight_layout(rect=[0, 0, 1, 0.98])  # [left, bottom, right, top]
        plt.savefig(savepath, bbox_inches = 'tight')
        plt.close()  


def plot_roc(config, y_test, y_score, cohort_suffix, axis=None, savepath = None):
    tumor_type = config['labels_to_use']

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    for i in range(len(tumor_type)):
        # print(i)
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['red', 'green', 'darkorange', 'darkviolet', 'dimgray', 'dodgerblue', 'gold'])


    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(8,8))

    for i, color, name in zip(range(len(tumor_type)), colors, tumor_type):
        axis.plot(fpr[i], tpr[i], color=color, lw=4, label='{0} (AUC = {1:0.3f})'.format(name, roc_auc[i]))

    axis.plot([0, 1], [0, 1], 'k--', lw=4)
    axis.set_xlim([0, 1.0])
    axis.set_ylim([0, 1.05])
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title('ROC [{}]'.format(cohort_suffix), fontsize=20)
    axis.legend(loc="best", prop=dict(size=18))

    if savepath is None: 
        return axis
    else:
        plt.savefig(savepath, bbox_inches = 'tight')
        plt.close()


def plot_precision_recall(config, y_test, y_score, cohort_suffix, axis=None, savepath = None):
    # Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-precision-recall-curve-for-each-class-and-iso-f1-curves
    
    tumor_type = config["labels_to_use"]


    # Calculate precision, recall for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(tumor_type)):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # setup plot details
    colors = cycle(['red', 'green', 'darkorange', 'darkviolet', 'dimgray', 'dodgerblue', 'gold'])

    lines = []
    labels = []

    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(8,8))

    # Plot the precision-recall for each class
    for i, color, name in zip(range(len(tumor_type)), colors, tumor_type):
        l, = axis.plot(recall[i], precision[i], color=color, lw=4)
        lines.append(l)
        labels.append('{0}, (AUC = {1:0.3f})'.format(name, average_precision[i]))

    # fig = axis.gcf()
    # fig.subplots_adjust(bottom=0.25)
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.05])
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.set_title('Precision-Recall curve [{}]'.format(cohort_suffix), fontsize=20)
    axis.legend(lines, labels, loc="best", prop=dict(size=18))

    if savepath is None: 
        return axis
    else:
        plt.savefig(savepath, bbox_inches = 'tight')
        plt.close()