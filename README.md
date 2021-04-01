# Programs for Segmenting Vegetation from Point Clouds

## Usage:

Before running the training program or reclassification program, ensure that you have pre-clipped two separate LAS or LAZ point clouds:

1. Point cloud containing points only of ***class A** (e.g. point cloud containing only vegetation points).
2. Point cloud containing points only of **class B** (e.g. point cloud containing only bare-Earth or non-vegetation points).

These point clouds can be segmented from larger point clouds using any number of programs. I utilize CloudCompare to manually clip out points for each class. It is important that each of the two segmented point clouds specified above include only points of the same class. Including points actually belonging to another class but included in a different sample point cloud will introduce error in the histogram values and will affect the computed Otsu's threshold value for each vegetation index.

First run the training program:
```
python veg_train.py
```
Then, run the reclassification program:
```
python veg_reclass.py
```
These programs compute the following vegetation indicies, their M-statistics, and Otsu threshold values:

- Excess Red (exr)
- Excess Green (exg)
- Excess Green-Red (exgr)
- Excess Blue (exb)
- Normal Green-Red Difference Index (ngrdi)
- Modified Green Red Vegetation Index (mgrvi)
- Green Leaf Index (gli)
- Red Green Blue Veggetation Index (rgbvi)
- Kawashima Index (ikaw)
- Green Lead Algorithm (gla)
- Visible Atmospherically Resistant Index (vari)*
- Woebbecke Index (wi)*
- Color Index of Vegetation Extraction (cive)*
- Vegetation (vega)*
- Combined Vegetation Index (com)*
  --> This index is a combination of (1) exg, (2) exgr, (3) cive, and (4) vega

*Asterisk denotes vegetation indicies that appear to be **unstable in simulated values** (i.e. their values are not properly constrained).*

Each of the vegetation indicies only requires some combination of the red, green, and blue color bands. No NIR, SWIR, or other band is required.
Citations for each of the vegetation indicies are included in the vegidx() function of the code.

The following Python modules are required:

```
os
ntpath
time
copy
subprocess
laspy
numpy
pandas
tKinter
```

## VEG_TRAIN.PY
### Inputs:

The program will automatically request the user to select 2 input files:

1. The point cloud containing vegetation points only (for training).
2. The point cloud containing only bare-Earth points (for training).

### Outputs:
An output CSV file will be generated with the following naming scheme:

> {veg_filename}\_{noveg_filename}.csv

Where *{veg_filename}* is the file name of the point cloud containing vegetation points only and *{noveg_filename}* is the name of the point cloud containing bare-Earth points only.

The output CSV will have the following attributes (columns) of information:

> {vegetation_index_name}     {minimum_possible_index_value}      {maximum_possible_index_value}      {M-statistic}       {Otsu_threshold_value}

## VEG_RECLASS.PY
### Inputs:

The program will automatically request the user to select 3 input files.

1. The point cloud to be reclassified.
2. The CSV file containing the vegetation index value ranges, M-statistics, and Otsu threshold values.
          --> This file is automatically created when you run veg_train.py

### Outputs:

A new LAZ file will be generated with the following naming scheme:
    
> {filename}\_reclass\_{vegetation_index_name}\_veg\_noveg.las

Where *{filename}* is the original point cloud file name and *{vegetation_index_name}* is the name of the vegetation index determined or selected to differentiate vegetation from bare-Earth using Otsu's thresholding approach.

The output LAZ file will be saved in the same directory as the input file and will contain all the original points with updated classification values corresponding to either vegetation or bare-Earth.


# FEEDBACK
**If you have any questions about how to implement the code, suggestions for improvements, or feedback, please leave a comment or report the issue with as much detail as possible.**
