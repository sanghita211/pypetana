# s-Petana: Compatibility with _scanner_ derived experimental data 

There are five stages to processing raw data to extract _area_, _perimeter_, _circularity_, _fractal dimension_ and the videos/images to use as inputs for Fricker Software [https://markfricker.org/77-2/software/physarum-network-analysis/].

## Stage 0

Download raw data with the following structure:

XXYYZZ/  
├── 1.zip  
├── 2.zip  
└── 3.zip  

where XX is year, YY is month and ZZ is day.

## Stage 1

Unzip the files using the following command:

```console
sh stage1-unzip.sh XXYYZZ
```

## Stage 2

Correct file extensions, convert the net pbm files to png (lossless) and remove the original net pbm files:

```console
sh stage2-fixjpgs.sh XXYYZZ
```

## Stage 3

Preprocess the png images. This will:

1. Detect the number and location of each petri dish, across all scanners.
2. Detect any masses (or objects other than biological organism) present, and use their location to calculate an angle of rotation to align all masses (or objects other than biological organism).
3. Segment, mask and rotate images, generating a new series of images for each petri dish in the `PREPROCESSED` folder.

The `PREPROCESSED` folder will have the following structure:

PREPROCESSED/  
├── XXYYZZ\_S\_EEEEE\_DD\_001.jpg  
├── ...  
└── XXYYZZ\_S\_EEEEE\_DD\_CCC.jpg  

Where S is scanner number (such as 1, 2 or 3), EEEEE is experiment, DD is petri dish (from 01 to 06, so on) and CCC is the last image in the specific series.

NOTE: _Provision for experimental data collection hiccup_: case of missing frames. In these cases the missing frame will be a copy of the last frame, meaning if you have frame 19 and 21 but are missing frame 20, a frame 20 will be generated as a copy of frame 19. Missing frames are marked with a red dot in the upper left corner of the image.

```console
python3 stage3-preprocess.py XXYYZZ
```

## Stage 4

Process the images of each series. This will (for each image in the series):

1. Detect and track the biological entity, drawing a contour around it in red.
2. Detect and track artifacts not part of the biological entity, drawing a contour around each in blue.
3. Calculate the _circularity_. This is calculated using OpenCV's area and perimeter methods.
4. Calculate _fractal dimension_ for box sizes `5`, and `10 to 200` in increments of `10` by:
  a. Creating a grid of boxes for the selected box size.
  b. Detecting if the box is partially or completely within the red contour, and if it is counting the box as an area box.
  c. Detecting if the box is partially touching the red contour, and if it is counting the box as a perimeter box.
  d. Use polyfit to fit the Log(BSR) vs Log(N), where BSR is box side length and N is the number of area boxes.
  e. Extract the slope to get the fractal dimension.
5. Write frame number, total area covered by the red contour, total perimeter of red contour, _circularity_ and _fractal dimension_ to a series-common output file in the `PROCESSED` folder.
6. Write frame number, box size, number of area boxes, and number of perimeter boxes to a separate series-common output file in the `PROCESSED` folder.
7. Generate a processed image to with blue and red contours, as well as a per-frame fractal dimension and circularity drawn on each image in the `PROCESSED` folder, marked with `_fracdim_` in the filename.
8. Generate a processed image where every pixel outside the red contour is masked, leaving only the biological entity (specific for use with Fricker software in case you have _Physarum_ growth) in the `PROCESSED` folder, marked with `_physarum_` in the filename.
9. Generate a plot of the fractal dimension fit for the processed image, marked with `_plt` in the filename.

The `PROCESSED` folder will have the following structure:

PROCESSED/  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD.dat  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD\_frac.dat  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD\_001.jpg  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD\_001\_plt.jpg  
├── ...  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD\_CCC.jpg  
├── nh\_fracdim\_XXYYZZ\_S\_EEEEE\_DD\_CCC\_plt.jpg  
├── nh\_physarum\_XXYYZZ\_S\_EEEEE\_DD\_001.jpg  
├── ...  
└── nh\_physarum\_XXYYZZ\_S\_EEEEE\_DD\_CCC.jpg  

Where the first `dat` file contains _circularity_ and _fractal dimension_ per frame, and the second `dat` file contains all fractal dimension data used in each fit per frame.

NOTE: It is important to target only a single series at a time with the following command. There are also options to modify threshold, changing the defaults. Details are not covered here. Check arxiv link:

```console
python3 stage4-process.py PREPROCESSED/XXYYZZ_S_EEEEE_DD_*.jpg
```

## Stage 5

Convert each series `_fracdim_` images and biological entity (if `_physarum_`) images into videos:

```console
sh stage5-convert.sh XXYYZZ
```
