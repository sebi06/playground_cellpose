import silence_tensorflow.auto
from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure, segmentation
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, black_tophat, disk, square, ball, closing, square, dilation
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_triangle, rank, median, gaussian
from skimage.segmentation import clear_border, watershed, random_walker
from skimage.color import label2rgb
from skimage.util import invert
from skimage.transform import rescale, resize
import scipy.ndimage as ndimage
from pylibCZIrw import czi as pyczi
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer as otw
import os
import sys
from tqdm import tqdm
from tqdm.contrib import itertools as it
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tifffile import imwrite, tiffcomment
from dataclasses import dataclass, field
import ome_types
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
import segmentation_tools as sgt


def erode_labels(segmentation, erosion_iterations, relabel=True):
    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = list()
    regions = regionprops(segmentation)

    def erode_mask(segmentation_labels, label_id, erosion_iterations, relabel=True):
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = ndimage.binary_erosion(only_current_label_id, iterations=erosion_iterations)

        if relabel:
            # relabeled_eroded = np.where(eroded == 1, label_id, 0)
            return (np.where(eroded == 1, label_id, 0))

        if not relabel:
            return (eroded)

    for i in range(len(regions)):
        label_id = regions[i].label
        list_of_eroded_masks.append(erode_mask(segmentation,
                                               label_id,
                                               erosion_iterations,
                                               relabel=relabel))

    # convert list of numpy arrays to stacked numpy array
    final_array = np.stack(list_of_eroded_masks)

    # max_IP to reduce the stack of arrays, each containing one labelled region, to a single 2D np array.
    final_array_labelled = np.sum(final_array, axis=0)

    return (final_array_labelled)


def expand5d(array):

    array = np.expand_dims(array, axis=-3)
    array = np.expand_dims(array, axis=-4)
    array5d = np.expand_dims(array, axis=-5)

    return array5d


class CziScaling:
    def __init__(self, filename: str, dim2none: bool = True) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        # get the XY scaling information
        try:
            self.X = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]) * 1000000
            self.Y = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][1]["Value"]) * 1000000
            self.X = np.round(self.X, 3)
            self.Y = np.round(self.Y, 3)
        except (KeyError, TypeError) as e:
            print("Error extracting XY Scale  :", e)
            self.X = 1.0
            self.Y = 1.0

        try:
            self.XUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["DefaultUnitFormat"]
            self.YUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][1]["DefaultUnitFormat"]
        except (KeyError, TypeError) as e:
            print("Error extracting XY ScaleUnit :", e)
            self.XUnit = None
            self.YUnit = None

        # get the Z scaling information
        try:
            self.Z = float(md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][2]["Value"]) * 1000000
            self.Z = np.round(self.Z, 3)
            # additional check for faulty z-scaling
            if self.Z == 0.0:
                self.Z = 1.0
            try:
                self.ZUnit = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][2]["DefaultUnitFormat"]
            except (IndexError, KeyError, TypeError) as e:
                print("Error extracting Z ScaleUnit :", e)
                self.ZUnit = self.XUnit
        except (IndexError, KeyError, TypeError) as e:
            print("Error extracting Z Scale  :", e)
            # set to isotropic scaling if it was single plane only
            self.Z = self.X
            self.ZUnit = self.XUnit

        # convert scale unit to avoid encoding problems
        if self.XUnit == "µm":
            self.XUnit = "micron"
        if self.YUnit == "µm":
            self.YUnit = "micron"
        if self.ZUnit == "µm":
            self.ZUnit = "micron"

        # get scaling ratio
        self.ratio = self.get_scale_ratio(scalex=self.X,
                                          scaley=self.Y,
                                          scalez=self.Z)

    @staticmethod
    def get_scale_ratio(scalex: float = 1.0,
                        scaley: float = 1.0,
                        scalez: float = 1.0) -> Dict:

        # set default scale factor to 1.0
        scale_ratio = {"xy": 1.0,
                       "zx": 1.0
                       }
        try:
            # get the factor between XY scaling
            scale_ratio["xy"] = np.round(scalex / scaley, 3)
            # get the scalefactor between XZ scaling
            scale_ratio["zx"] = np.round(scalez / scalex, 3)
        except (KeyError, TypeError) as e:
            print(e, "Using defaults = 1.0")

        return scale_ratio


def get_channel_from_czi(czifile: str,
                         ch_index: int = 0,
                         roi=None) -> (np.ndarray):

    with pyczi.open_czi(czifile) as czidoc:

        # get specific planes
        img = czidoc.read(plane={'C': ch_index},
                          roi=roi,
                          scene=0)[..., 0]

    return img


def create_labeldata(int_image: np.ndarray,
                     labels: np.ndarray,
                     num_erode: int = 1,
                     use_watershed: bool = True,
                     min_distance=5,
                     relabel: bool = False,
                     verbose=False) -> Tuple[np.ndarray, np.ndarray]:

    if use_watershed:
        #new_labels = sgt.apply_watershed(new_labels, min_distance=min_distance)

        new_labels = sgt.apply_watershed_adv(np.squeeze(int_image),
                                             labels,
                                             filtermethod_ws='median',
                                             filtersize_ws=7,
                                             min_distance=min_distance,
                                             radius=7)

        #new_labels = (new_labels >= 1).astype(np.uint8)

    new_labels = erode_labels(new_labels, num_erode, relabel=relabel)
    new_labels = expand5d(new_labels).astype(np.uint8) * 255

    if verbose:
        print("New labels info: ", new_labels.min(), new_labels.max(), new_labels.shape, new_labels.dtype)

    # convert to desired type
    # new_labels = new_labels.astype(np.uint8) * 255
    background = invert(new_labels)

    return np.squeeze(new_labels), np.squeeze(background)


def show_plot(img: np.ndarray, labels: np.ndarray, new_labels: np.ndarray) -> None:

    # show the results
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input")
    #ax[1].imshow(render_label(labels.astype(np.uint16), img=img))
    #ax[1].set_title("pred + input")
    #ax[1].imshow(render_label(new_labels, img=img))
    #ax[1].set_title("pred + input + erode")
    ax[1].imshow(new_labels)
    ax[1].set_title("pred + erode")
    plt.show()


def save_OMETIFF(img_FL: np.ndarray,
                 img_TL: np.ndarray,
                 new_labels: np.ndarray,
                 background: np.ndarray,
                 savepath_FL: str = "DAPI.ome.tiff",
                 savepath_TL: str = "PGC.ome.tiff",
                 savepath_NUC: str = "PGC_nuc.ome.tiff",
                 savepath_BGRD: str = "PGC_background.ome.tiff",
                 pixels_physical_sizes: List[float] = [1.0, 1.0, 1.0],
                 channel_names: Dict[str, str] = {"FL": "FL", "TL": "TL", "NUC": "NUC", "BGRD": "BGRD"}) -> None:

    # # write the array as an OME-TIFF incl. the metadata for the labels
    # otw.OmeTiffWriter.save(expand5d(img_FL), savepath_FL,
    #                        channel_names=channel_names["FL"],
    #                        pixels_physical_sizes=pixels_physical_sizes,
    #                        dim_order="TZCYX")

    # # write the array as an OME-TIFF incl. the metadata for the labels
    # otw.OmeTiffWriter.save(expand5d(img_TL), savepath_TL,
    #                        channel_names=channel_names["TL"],
    #                        pixels_physical_sizes=pixels_physical_sizes,
    #                        dim_order="TZCYX")

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_FL, savepath_FL,
                           channel_names=channel_names["FL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_TL, savepath_TL,
                           channel_names=channel_names["TL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the label
    otw.OmeTiffWriter.save(new_labels, savepath_NUC,
                           channel_names=channel_names["NUC"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the background
    otw.OmeTiffWriter.save(background, savepath_BGRD,
                           channel_names=["BGRD"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")


def segment_nuclei_stardist(image2d, sdmodel,
                            prob_thresh=0.5,
                            overlap_thresh=0.3,
                            overlap_label=None,
                            norm_pmin=1.0,
                            norm_pmax=99.8,
                            norm_clip=False):

    # normalize image
    image2d_norm = normalize(image2d,
                             pmin=norm_pmin,
                             pmax=norm_pmax,
                             axis=None,
                             clip=norm_clip,
                             eps=1e-20,
                             dtype=np.float32)

    # predict the instances of th single nuclei
    mask2d, details = sdmodel.predict_instances(image2d_norm,
                                                axes=None,
                                                normalizer=None,
                                                prob_thresh=prob_thresh,
                                                nms_thresh=overlap_thresh,
                                                n_tiles=None,
                                                show_tile_progress=True,
                                                overlap_label=overlap_label,
                                                verbose=False)

    return mask2d


def area_filter(im: np.ndarray, area_min: int = 10, area_max: int = 100000) -> np.ndarray:
    """
    Filters objects in an image based on their areas.

    Parameters
    ----------
    im : 2d-array, int
        Labeled segmentation mask to be filtered. 
    area_bounds : tuple of ints
        Range of areas in which acceptable objects exist. This should be 
        provided in units of square pixels.

    Returns
    -------
    im_relab : 2d-array, int
        The relabeled, filtered image.
    """

    # Extract the region props of the objects.
    props = measure.regionprops(im)

    # Extract the areas and labels.
    areas = np.array([prop.area for prop in props])
    labels = np.array([prop.label for prop in props])

    # Make an empty image to add the approved cells.
    im_approved = np.zeros_like(im)

    # Threshold the objects based on area and eccentricity
    for i, _ in enumerate(areas):
        if areas[i] > area_min and areas[i] < area_max:
            im_approved += im == labels[i]

    # Relabel the image.
    im_filt = measure.label(im_approved > 0)

    return im_filt


##########################################################################
plot = False

basefolder = r"data"
dir_FL = os.path.join(basefolder, "fluo")
dir_LABEL = os.path.join(basefolder, "label")
dir_TL = os.path.join(basefolder, "trans")

os.makedirs(dir_FL, exist_ok=True)
os.makedirs(dir_LABEL, exist_ok=True)
os.makedirs(dir_TL, exist_ok=True)

suffix_orig = ".ome.tiff"
suffix_NUC = "_nuc.ome.tiff"
suffix_BGRD = "_background.ome.tiff"
use_tiles = False
target_scaleXY = 0.5
tilesize = 400
num_erode = 1
rescale = False

# prints a list of available models
# StarDist2D.from_pretrained()
model = StarDist2D.from_pretrained('2D_versatile_fluo')
stardist_prob_thresh = 0.5
stardist_overlap_thresh = 0.2
stardist_overlap_label = None  # 0 is not supported yet
stardist_norm = True
stardist_norm_pmin = 1
stardist_norm_pmax = 99.8
stardist_norm_clip = False
area_min = 20
area_max = 1000000

use_watershed = True
min_distance = 100

ch_id_FL = 0
ch_id_TL = 1

ext = ".czi"

# iterating over all files
for file in os.listdir(basefolder):
    if file.endswith(ext):

        print("Processing CZI file:", file)

        cziname = file
        cziname_NUC = file[:-4] + "_onlyFL"
        cziname_TL = file[:-4] + "_onlyTL"

        # get the scaling from the CZI
        cziscale = CziScaling(os.path.join(basefolder, cziname))
        pixels_physical_sizes = [1, cziscale.X, cziscale.Y]
        scale_forward = target_scaleXY / cziscale.X
        new_shapeXY = int(np.round(tilesize * scale_forward, 0))

        # open a CZI instance to read and in parallel one to write
        with pyczi.open_czi(os.path.join(basefolder, file)) as czidoc_r:

            if use_tiles:

                tilecounter = 0

                # create a "tile" by specifying the desired tile dimension and minimum required overlap
                tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=tilesize,
                                                                  total_tile_height=tilesize,
                                                                  min_border_width=8)

                # get the size of the bounding rectangle for the scene
                tiles = tiler.tile_rectangle(czidoc_r.scenes_bounding_rectangle[0])

                # show the created tile locations
                for tile in tiles:
                    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

                # loop over all tiles created by the "tiler"
                for tile in tqdm(tiles):

                    # read a specific tile from the CZI using the roi parameter
                    tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))
                    tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))

                    if rescale:
                        # scale the FL image to 0.5 micron per pixel (more or less)
                        tile2d_FL_scaled = resize(tile2d_FL, (new_shapeXY, new_shapeXY), preserve_range=True, anti_aliasing=True)

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels_scaled = segment_nuclei_stardist(tile2d_FL_scaled, model,
                                                                prob_thresh=stardist_prob_thresh,
                                                                overlap_thresh=stardist_overlap_thresh,
                                                                overlap_label=stardist_overlap_label,
                                                                # norm=stardist_norm,
                                                                norm_pmin=stardist_norm_pmin,
                                                                norm_pmax=stardist_norm_pmax,
                                                                norm_clip=stardist_norm_clip)

                        # scale the label image back to the original size preserving the label values
                        labels = resize(labels_scaled, (tilesize, tilesize), anti_aliasing=False, preserve_range=True).astype(int)

                    if not rescale:

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels = segment_nuclei_stardist(tile2d_FL, model,
                                                         prob_thresh=stardist_prob_thresh,
                                                         overlap_thresh=stardist_overlap_thresh,
                                                         overlap_label=stardist_overlap_label,
                                                         # norm=stardist_norm,
                                                         norm_pmin=stardist_norm_pmin,
                                                         norm_pmax=stardist_norm_pmax,
                                                         norm_clip=stardist_norm_clip)

                    # filter labels by size
                    labels = area_filter(labels, area_min=area_min, area_max=area_max)

                    new_labels, background = create_labeldata(tile2d_FL, labels,
                                                              num_erode=num_erode,
                                                              watershed=use_watershed,
                                                              min_distnace=min_distance,
                                                              relabel=False,
                                                              verbose=False)

                    show_plot(tile2d_FL, labels, np.squeeze(new_labels))

                    # save the original FL channel as OME-TIFF
                    savepath_FL = os.path.join(dir_FL, cziname_NUC + "_t" + str(tilecounter) + suffix_orig)

                    # save the original TL (PGC etc. ) channel as OME_TIFF
                    savepath_TL = os.path.join(dir_TL, cziname_TL + "_t" + str(tilecounter) + suffix_orig)

                    # save the labels for the nucleus and the background as OME-TIFF
                    savepath_BGRD = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_BGRD)
                    savepath_NUC = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_NUC)

                    # save the OME-TIFFs
                    save_OMETIFF(tile2d_FL, tile2d_TL, new_labels, background,
                                 savepath_FL=savepath_FL,
                                 savepath_TL=savepath_TL,
                                 savepath_NUC=savepath_NUC,
                                 savepath_BGRD=savepath_BGRD,
                                 pixels_physical_sizes=pixels_physical_sizes)

                    tilecounter += 1

            if not use_tiles:

                # read a specific tile from the CZI using the roi parameter
                tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL})[..., 0]
                tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL})[..., 0]

                # get the prediction for the current tile
                # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                labels = segment_nuclei_stardist(tile2d_FL, model,
                                                 prob_thresh=stardist_prob_thresh,
                                                 overlap_thresh=stardist_overlap_thresh,
                                                 overlap_label=stardist_overlap_label,
                                                 # norm=stardist_norm,
                                                 norm_pmin=stardist_norm_pmin,
                                                 norm_pmax=stardist_norm_pmax,
                                                 norm_clip=stardist_norm_clip)

                # filter labels by size
                labels = area_filter(labels, area_min=area_min, area_max=area_max)

                new_labels, background = create_labeldata(tile2d_FL, labels,
                                                          num_erode=num_erode,
                                                          use_watershed=use_watershed,
                                                          min_distance=min_distance,
                                                          relabel=False,
                                                          verbose=False)

                show_plot(tile2d_FL, labels, new_labels)

                # save the original FL channel as OME-TIFF
                savepath_FL = os.path.join(dir_FL, cziname_NUC[:-4] + suffix_orig)

                # save the original TL (PGC etc. ) channel as OME_TIFF
                savepath_TL = os.path.join(dir_TL, cziname_TL[:-4] + suffix_orig)

                # save the labels for the nucleus and the background as OME-TIFF
                savepath_BGRD = os.path.join(dir_LABEL, cziname_TL[:-4] + suffix_BGRD)
                savepath_NUC = os.path.join(dir_LABEL, cziname_TL[:-4] + suffix_NUC)

                # save the OME-TIFFs
                save_OMETIFF(tile2d_FL, tile2d_TL, new_labels, background,
                             savepath_FL=savepath_FL,
                             savepath_TL=savepath_TL,
                             savepath_NUC=savepath_NUC,
                             savepath_BGRD=savepath_BGRD,
                             pixels_physical_sizes=pixels_physical_sizes)

    else:
        continue

print("Done.")

# # write the array as an OME-TIFF incl. the metadata for the labels
# otw.OmeTiffWriter.save(new_labels, savepath_label,
#                        channel_names=["nuc"],
#                        pixels_physical_sizes=pixels_physical_sizes,
#                        dim_order="TZCYX")

# # write the array as an OME-TIFF incl. the metadata for the background
# otw.OmeTiffWriter.save(background, savepath_bgrd,
#                        channel_names=["bgrd"],
#                        pixels_physical_sizes=pixels_physical_sizes,
#                        dim_order="TZCYX")


# imwrite(savepath2_label,
#         label5d,
#         bigtiff=True,
#         photometric="minisblack",
#         metadata={
#             'axes': dim_order,
#             'SignificantBits': 8,
#             'Pixels': {
#                 'PhysicalSizeX': cziscale.X,
#                 'PhysicalSizeXUnit': 'µm',
#                 'PhysicalSizeY': cziscale.Y,
#                 'PhysicalSizeYUnit': 'µm'
#             },
#         },
#         )

# ome_xml = tiffcomment(savepath2_label)
# ome = ome_types.from_xml(ome_xml)
# ome.images[0].description = 'Nucleus Labels'
# ome_xml = ome.to_xml()
# tiffcomment(savepath2_label, ome_xml)

# imwrite(savepath2_bgrd,
#         background,
#         bigtiff=True,
#         photometric="minisblack",
#         metadata={
#             'axes': dim_order,
#             'SignificantBits': 8,
#             'Pixels': {
#                 'PhysicalSizeX': cziscale.X,
#                 'PhysicalSizeXUnit': 'µm',
#                 'PhysicalSizeY': cziscale.Y,
#                 'PhysicalSizeYUnit': 'µm'
#             },
#         },
#         )

# ome_xml = tiffcomment(savepath2_bgrd)
# ome = ome_types.from_xml(ome_xml)
# ome.images[0].description = 'Nucleus Labels'
# ome_xml = ome.to_xml()
# tiffcomment(savepath2_bgrd, ome_xml)
