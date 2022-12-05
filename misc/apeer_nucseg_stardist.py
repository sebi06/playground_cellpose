# -*- coding: utf-8 -*-

#################################################################
# File        : apeer_nucseg_stardist.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#################################################################

import os
import numpy as np
import pandas as pd
from skimage import measure, segmentation
from skimage.measure import regionprops
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
from tqdm.contrib.itertools import product
import segmentation_tools as sgt
import segmentation_stardist as sg_sd
from stardist.models import StarDist2D
from typing import List, Dict, NamedTuple, Tuple, Optional, Type, Any, Union
import matplotlib.pyplot as plt


def execute(image_path: str,
            chindex_nucleus: int = 0,
            sd_modelbasedir: str = "stardist_models",
            sd_modelfolder: str = "2D_versatile_Fluo",
            prob_th: float = 0.5,
            ov_th: float = 0.3,
            do_area_filter: bool = True,
            minsize_nuc: int = 20,
            maxsize_nuc: int = 5000,
            blocksize: int = 1024,
            min_overlap: int = 128,
            n_tiles: Optional[int] = None,
            norm_pmin: int = 1,
            norm_pmax: float = 99.8,
            norm_clip: bool = False,
            verbose: bool = True,
            do_clear_borders: bool = True,
            flatten_labels: bool = False,
            normalize_whole: bool = True) -> Tuple[str, str, str]:
    """This function segments a CZI image plane-by-plane using the specified StarDist2D model.

    Args:
        image_path (str): Filepath of CZI image to be processed
        chindex_nucleus (int, optional): Channel index containing the cell nuclei. Defaults to 0.
        sd_modelbasedir (str, optional): Path to folder containing StarDist models. Defaults to "stardist_models".
        sd_modelfolder (str, optional): Path to subfolder for a specific StarDist model. Defaults to "2D_versatile_fluo".
        prob_th (float, optional): Probability threshold. Defaults to 0.5.
        ov_th (float, optional): Overlap threshold. Defaults to 0.3.
        do_area_filter (bool, optional): Filter nuclei by size. Defaults to True.
        minsize_nuc (int, optional): Minimum nucleus size [pixel]. Defaults to 20.
        maxsize_nuc (int, optional): Maximum nucleus size [pixel]. Defaults to 5000.
        blocksize (int, optional): Blocksize when segmenting large images. Defaults to 1024.
        min_overlap (int, optional): Minimum overlap between blocks. Defaults to 128.
        n_tiles (Optional[int], optional): Tiling. Defaults to None.
        norm_pmin (int, optional): Percentile Minimum for normalization. Defaults to 1.
        norm_pmax (float, optional): Percentile Maximum for normalization. Defaults to 99.8.
        norm_clip (bool, optional): Clip normalization. Defaults to False.
        verbose (bool, optional): Show additional output. Defaults to True.
        do_clear_borders (bool, optional): Option to remove objects from image borders. Defaults to True.
        flatten_labels (bool, optional): Specify if all labels should have the same pixel value. Defaults to False.
        local_normalize (bool, optional): Use local normalization instead of overall calculation (slow for large images). Defaults to True.

    Returns:
        Tuple[str, str, str]: Tuple containing the filenames of the created outputs.
    """

    image_name = os.path.basename(image_path)
    print('Current Image: ', image_name)
    savepath = misc.get_fname_woext(image_name) + "_segSD.czi"

    # get the metadata
    mdata = czimd.CziMetadata(image_path)

    # check the channel number for the nucleus
    if mdata.image.SizeC is not None:
        if chindex_nucleus + 1 > mdata.image.SizeC and mdata.isRGB is False:
            print('Selected Channel for nucleus does not exist. Use channel = 1.')
            chindex_nucleus = 0
    if mdata.image.SizeC is None:
        chindex_nucleus = 0

    # define columns names for dataframe
    cols = ["WellId", "Well_ColId", "Well_RowId", "S", "T", "Z", "C", "Number"]
    objects = pd.DataFrame(columns=cols)
    results = pd.DataFrame()

    # measure region properties
    to_measure = ('label',
                  'area',
                  'centroid',
                  'max_intensity',
                  'mean_intensity',
                  'min_intensity',
                  'bbox')

    units = ["micron**2", "pixel", "pixel", "cts", "counts", "cts", ]

    # read model from folder or the internet
    sd_model = StarDist2D(None, name=sd_modelfolder, basedir=sd_modelbasedir)

    # use this line to read the model directly from the internet repo
    # sd_model = StarDist2D.from_pretrained('2D_versatile_fluo')

    with pyczi.create_czi(savepath, exist_ok=True) as czidoc_w:

        with pyczi.open_czi(image_path) as czidoc_r:

            # check if dimensions are None (because the do not exist for that image)
            sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
            sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
            sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
            sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

            # read array for the scene
            for s, t, z, in product(range(sizeS),
                                    range(sizeT),
                                    range(sizeZ)):

                values = {'S': s, 'T': t, 'Z': z, 'C': chindex_nucleus, 'Number': 0}

                # read 2D plane in case there are (no) scenes
                if mdata.image.SizeS is None:
                    # img2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': chindex_nucleus})  # [..., 0]
                    img2d = czidoc_r.read(plane={'T': t, 'Z': z, 'C': chindex_nucleus})  # [..., 0]
                else:
                    img2d = czidoc_r.read(
                        plane={'T': t, 'Z': z, 'C': chindex_nucleus}, scene=s)  # [..., 0]

                if not mdata.isRGB:
                    img2d = img2d[..., 0]
                    axes = "YX"

                if mdata.isRGB:
                    axes = "YXC"

                dtype_orig = img2d.dtype

                labels = sg_sd.segment_nuclei_stardist(img2d, sd_model,
                                                       axes=axes,
                                                       prob_thresh=prob_th,
                                                       overlap_thresh=ov_th,
                                                       overlap_label=None,
                                                       # blocksize=blocksize,
                                                       min_overlap=min_overlap,
                                                       n_tiles=n_tiles,
                                                       norm_pmin=norm_pmin,
                                                       norm_pmax=norm_pmax,
                                                       norm_clip=norm_clip,
                                                       normalize_whole=normalize_whole)

                print("StarDist - OBjects:", labels.max())

                # find boundaries
                bound = segmentation.find_boundaries(
                    labels, connectivity=2, mode="thicker", background=0)

                # set all boundary pixel inside label image = Zero by inverting the boundary image
                labels = labels * ~bound

                # show_plot(bound, ~bound, labels)

                if do_area_filter:
                    # filter labels by size
                    # labels, num_labels = sgt.area_filter(labels,
                    #                                     area_min=minsize_nuc,
                    #                                     area_max=maxsize_nuc)

                    labels, num_labels = sgt.filter_labels(labels, minsize_nuc, maxsize_nuc)

                    print("Area Filter - Objects:", num_labels)

                if do_clear_borders:
                    # clear border objects
                    labels = segmentation.clear_border(labels)
                    print("Clear Borders - Objects:", labels.max())

                # measure the specified parameters store in dataframe
                props = pd.DataFrame(measure.regionprops_table(labels,
                                                               intensity_image=img2d,
                                                               properties=to_measure)
                                     ).set_index('label')

                # add well information for CZI metadata
                try:
                    props['WellId'] = mdata.sample.well_array_names[s]
                    props['Well_ColId'] = mdata.sample.well_colID[s]
                    props['Well_RowId'] = mdata.sample.well_rowID[s]
                except (IndexError, KeyError) as error:
                    print('Error:', error)
                    print('Well Information not found. Using S-Index.')
                    props['WellId'] = s
                    props['Well_ColId'] = s
                    props['Well_RowId'] = s

                # add plane indices
                props['S'] = s
                props['T'] = t
                props['Z'] = z
                props['C'] = chindex_nucleus

                values = {"WellId": props['WellId'],
                          "Well_ColId": props['Well_ColId'],
                          "Well_RowId": props['Well_RowId'],
                          "S": s,
                          "T": t,
                          "Z": z,
                          "C": chindex_nucleus,
                          "Number": props.shape[0]}

                # count the number of objects
                # values['Number'] = props.shape[0]

                if verbose:
                    print('Well:', props['WellId'].iloc[0], ' Objects: ', values['Number'])

                # update dataframe containing the number of objects
                # objects = objects.append(pd.DataFrame(values, index=[0]), ignore_index=True)
                objects = pd.concat([objects, pd.DataFrame(values, index=[0])], ignore_index=True)

                # results = results.append(props, ignore_index=True)
                results = pd.concat([results, props], ignore_index=True)

                # add dimension for CZI pixel type at the end of array - [Y, X, 1]
                labels = labels[..., np.newaxis]

                # convert to desired dtype in place
                labels = labels.astype(dtype_orig, copy=False)

                if flatten_labels:
                    labels[labels > 0] = 255

                    # write the label image to CZI
                if mdata.image.SizeS is None:

                    # write 2D plane in case of no scenes
                    czidoc_w.write(labels, plane={"T": t,
                                                  "Z": z,
                                                  "C": chindex_nucleus})
                else:
                    # write 2D plane in case scenes exist
                    czidoc_w.write(labels, plane={"T": t,
                                                  "Z": z,
                                                  "C": chindex_nucleus},
                                   scene=s,
                                   location=(czidoc_r.scenes_bounding_rectangle[s].x,
                                             czidoc_r.scenes_bounding_rectangle[s].y)
                                   )

        # write scaling the the new czi and check if valid scaling exists
        if mdata.scale.X is None:
            mdata.scale.X = 1.0
        if mdata.scale.Y is None:
            mdata.scale.Y = 1.0
        if mdata.scale.Y is None:
            mdata.scale.Z = 1.0

        czidoc_w.write_metadata(document_name=savepath,
                                channel_names={0: "nucSEG"},
                                scale_x=mdata.scale.X * 10**-6,
                                scale_y=mdata.scale.Y * 10**-6,
                                scale_z=mdata.scale.Z * 10**-6)

    # reorder dataframe with single objects
    new_order = list(results.columns[-7:]) + list(results.columns[:-7])
    results = results.reindex(columns=new_order)

    # define name for CSV tables
    obj_csv = misc.get_fname_woext(image_name) + '_obj.csv'
    objparams_csv = misc.get_fname_woext(image_name) + '_objparams.csv'

    # save the DataFrames as CSV tables
    objects.to_csv(obj_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Table as CSV :', obj_csv)

    results.to_csv(objparams_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Parameters Table as CSV :', objparams_csv)
    print('Segmentation done.')

    return (obj_csv, objparams_csv, savepath)


def show_plot(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray) -> None:

    # show the results
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    ax[0].imshow(img1, cmap="gray")
    ax[1].imshow(img2, cmap="gray")
    ax[2].imshow(img3, cmap="gray")
    plt.show()


# Test Code locally
if __name__ == "__main__":

    #filename = r'input/A01.czi'
    filename = r"F:\Testdata_Zeiss\CZI_Testfiles\Tumor_H+E_small2.czi"
    # filename = r'input/DAPI_20x.czi'
    #filename = r'input/WP384_4Pos_B4-10_DAPI.czi'
    # filename = r'input/Image2_A3_01_1_1_DAPI_001.czi'
    # filename = r'input\Osteosarcoma_01.czi'

    # only FL-stained nuclei are supported
    modeltype = '2D_versatile_fluo'
    #modeltype = "2D_versatile_he"
    #modeltype = "2D_dsb2018_fluo"

    outputs = execute(filename,
                      chindex_nucleus=0,
                      sd_modelbasedir='stardist_models',
                      sd_modelfolder=modeltype,
                      prob_th=0.5,
                      ov_th=0.3,
                      do_area_filter=True,
                      minsize_nuc=200,
                      maxsize_nuc=1000,
                      # blocksize=4096,
                      min_overlap=128,
                      n_tiles=3,
                      norm_pmin=1,
                      norm_pmax=99.8,
                      norm_clip=False,
                      verbose=True,
                      do_clear_borders=True,
                      flatten_labels=True,
                      normalize_whole=True)

    print(outputs[0])
    print(outputs[1])
    print(outputs[2])
