import th_modules.sam_segmenter as sam_seg
import th_modules.unet_segmenter as unet_seg
import th_modules.nf1_reader as nf1_reader
import th_modules.mask_analysis as mask_analysis

import matplotlib.pyplot as plt
#from matplotlib.widgets import RectangleSelector

import numpy as np
#import cv2


'''APPLICATION PARAMETERS'''
use_demo_images = True
unet_model_path = "./HAM10000 UNET/test/unet_SKIN_32x32_l0.1562_a0.9363.h5"
nf1_source_path = "./nf1_database/october/"
double_pass = False # whether to run the algorithm twice at 32 and 64 patch sizes, and average results
patch_size = 32
patch_step_single_pass = 0.5
sam_checkpoint_path = "./sam_vith_model.pth"
num_images = 10#int(input("How Many Images to Process: "))

""" Stage 1: SAM
    - Importing SAM
    - Initialising instance of SAM
"""
print("Loading SAM...")

#from segment_anything import SamPredictor

bg_predictor = sam_seg.init_sam_model(checkpoint_path=sam_checkpoint_path)


""" Importing NF1 Data 
    - reading paths from /skin_masks/ file tree.
    - reading images with cv2 in either colour or binary, depending on image or mask.
    - resizing and cropping to standard sizes that maximise mask area within the image.
"""
print("Loading NF1 Data...")

nf1_images = nf1_reader.get_nf1_images(nf1_source_path,num_images,demo=use_demo_images)
num_images = len(nf1_images)
""" Stage 2: U-NET
    - Initialise a U-NET insance
    - Compile
    - Load weights from previously trained instance.
"""
print("Loading Tumour Segmenter...")

network_input_size = (
    patch_size,
    patch_size,
    3,
) 

tm_predictor = unet_seg.init_unet_model(
    input_size=network_input_size,
    weights_path=unet_model_path,
)

""" Stage 3: Prompting User for Specification of Image Foreground
"""

for i in range(0, num_images):
    nf1_im = nf1_images[i].astype(np.uint8)
    
    print("Encoding Image...")
    bg_predictor.set_image(nf1_im)
    retry = True
    print("Prompting for Foreground Specification...")

    while retry:
        masks, scores, _, box, coords, labels = sam_seg.get_masks_from_input(
            nf1_im, bg_predictor, multimask=False
        )
        for mask, score in zip(masks, scores):
            sam_seg.show_sam_output(nf1_im, mask, score, coords, labels, box)


        retry = input("Retry Mask? (y/[n]) ")
        if retry == 'y':
            retry = True
        else: 
            retry = False

    mask = sam_seg.mask_to_binary(mask)
    plt.imshow(mask,cmap='gray')
    plt.show()


    mask_im = nf1_reader.apply_mask(nf1_im,mask) 
    plt.imshow(mask_im)
    plt.show()

    """ Stage 4: Generating Patch Tumour Masks
    """

    print("Generating NF1 Patch Predictions...")
    

    if double_pass:
        _, _, _, recon_im_32, _ = unet_seg.analyse_n_patches_flat(
            imag=nf1_im,
            mask=mask,
            model=tm_predictor,
            network_size=network_input_size,
            binary_threshold=0.5,
            n_ims_max=-1,
            patch_size=patch_size,
            patch_step=1,
        )
        _, _, _, recon_im_64, _ = unet_seg.analyse_n_patches_flat(
            imag=nf1_im,
            mask=mask,
            model=tm_predictor,
            network_size=network_input_size,
            binary_threshold=0.5,
            n_ims_max=-1,
            patch_size=2*patch_size,
            patch_step=1,
        )
        r32_op =unet_seg.morph_open_image(recon_im_32)
        r64_op = unet_seg.morph_open_image(recon_im_64)
        recon_im = ( r32_op + r64_op) * 0.5
    else:
        _, _, _, recon_im_32, _ = unet_seg.analyse_n_patches_flat(
        imag=nf1_im,
        mask=mask,
        model=tm_predictor,
        network_size=network_input_size,
        binary_threshold=0.5,
        n_ims_max=-1,
        patch_size=patch_size,
        patch_step=patch_step_single_pass,
        )
        recon_im = unet_seg.morph_open_image(recon_im_32)
    
        binary =unet_seg.imbinarise(recon_im,thresh=(1.0 / patch_step_single_pass) - 0.1)
    

    '''mask analysis for NF1 severity'''

    print("Analysing Tumour Segmentation...")

    tumour_skin_perc = mask_analysis.tumour_density(mask,binary)
    print(f"Tumours Make up {100*tumour_skin_perc:.4f}% of Skin Area.")

    num_clusters,cluster_areas = mask_analysis.cluster_stats(binary)
    print(f"{num_clusters} Tumour Clusters Detected (Note that a single cluster may contain multiple adjacent NFs).")


    if double_pass:
        sam_seg.figure_mask_image(mask_im,
                                mask=r32_op,
                                title="Result, 32-Bit Patches")
        
        sam_seg.figure_mask_image(mask_im,
                                mask=r64_op,
                                title="Result, 64-Bit Patches")
        
        sam_seg.figure_mask_image(mask_im,
                                mask=recon_im,
                                title="Intersection of 32, 64-Bit Passes")
    #else: 
        #sam_seg.figure_mask_image(mask_im,mask=recon_im,title="Tumour Segmentation Result")
    
    sam_seg.figure_mask_image(mask_im,
                              mask=binary,
                              title="Mask Overlaid on Image")
    
    sam_seg.figure_mask_image(mask_im, title = "Image")
    sam_seg.figure_mask_image(mask, title = "Skin Mask")
    sam_seg.figure_mask_image(binary, title = "Tumour Mask")

    #nf1_reader.mask_to_bounding_boxes(nf1_im, binary,title=f"{num_clusters} Tumour Clusters Detected")

    mask_analysis.areas_to_hist(cluster_areas,n_bins=100,n_clusters=num_clusters,perc=100*tumour_skin_perc)

    plt.show()


    

