# Programs for Segmenting Vegetation from Point Clouds

These programs are designed to segment vegetation from bare-Earth points in a dense point cloud, although they may also be used to segment any two classes that are visually distinguishable from each other. There are two approaches:

1. The non-machine learning approach utilizes a vegetation index and Otsu's thresholding method. This approach is more computationally demanding and the appropriate vegetation index and threshold value are likely to vary by location and application.
2. The machine learning approach utilizes the Tensorflow API. This approach is more efficient, less subjective, and more robust across geographies and applications. Although it is faster with access to a GPU, a CPU can also be used. *(Note: The code is not optimized for use with multiple GPUs.)*

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

Each of the vegetation indices only requires some combination of the red, green, and blue color bands. No NIR, SWIR, or other band is required.
Citations for each of the vegetation indices are included in the vegidx() function of the code.

The following Python modules are required:

```
os
ntpath
time
copy
subprocess
laspy
lasrs *(required to write compressed LAZ file with laspy)*
numpy
pandas
tKinter
tensorflow (or tensorflow-gpu) **Only if using the machine learning approach**
```

## Installation:

Clone the repository locally first.
```
git clone https://github.com/pwernette/point_cloud_vegetation_filtering
```

Then, create a virtual environment from one of the .yml environment files in the environment sub-directory.

To create an environment for the non-machine learning approach (utilizing Otsu's thresholding method and a vegetation index), create the environment using:
```
conda env create -f PC_veg_filter_env.yml
```

To create an environment for the machine learning approach (utilizing Tensorflow), create the environment using:
```
conda env create -f ML_veg_filter_env.yml
```

Once you have created the virtual environment, activate the environment by either:
```
conda activate vegfilter
```
or
```
conda activate mlvegfilter
```

## Usage (No Machine Learning):

Before running the training program or reclassification program, ensure that you have pre-clipped two separate LAS or LAZ point clouds:

1. Point cloud containing points only of **class A** (e.g. point cloud containing only vegetation points).
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

### veg_train.py
#### Inputs:

The program will automatically request the user to select 2 input files:

1. The point cloud containing vegetation points only (for training).
2. The point cloud containing only bare-Earth points (for training).

#### Outputs:
An output CSV file will be generated with the following naming scheme:

> {veg_filename}\_{noveg_filename}.csv

Where *{veg_filename}* is the file name of the point cloud containing vegetation points only and *{noveg_filename}* is the name of the point cloud containing bare-Earth points only.

The output CSV will have the following attributes (columns) of information:

> {vegetation_index_name}     {minimum_possible_index_value}      {maximum_possible_index_value}      {M-statistic}       {Otsu_threshold_value}

### veg_reclass.py
#### Inputs:

The program will automatically request the user to select 2 input files.

1. The point cloud to be reclassified.
2. The CSV file containing the vegetation index value ranges, M-statistics, and Otsu threshold values.
          --> This file is automatically created when you run veg_train.py

#### Outputs:

A new LAZ file will be generated with the following naming scheme:

> {filename}\_reclass\_{vegetation_index_name}\_veg\_noveg.laz

Where *{filename}* is the original point cloud file name and *{vegetation_index_name}* is the name of the vegetation index determined or selected to differentiate vegetation from bare-Earth using Otsu's thresholding approach.

The output LAZ file will be saved in the same directory as the input file and will contain all the original points with updated classification values corresponding to either vegetation or bare-Earth.

## Usage (Machine Learning):

The machine learning approach can be run (1) as two separate programs, one for ML model training and a second for LAS/LAZ file (re)classification, or (2) as a single program that builds and trains a ML model and then uses that model to reclassify a LAS/LAZ file.

Command line options are available to for both the two program and one program options to cut down on pop-up windows and aid in batch scripting:
| Option | Option type(s) | Default value(s) | Option description/function(s) |
| --- | --- | --- | --- |
| `-v`, `-veg` | string | NA | Point cloud containing vegetation points only |
| `-g`, `-ground` | string | NA | Point cloud containing ground points only |
| `-m`, `-name` | string | NA | ML model name |
| `-vi`, `-index` | string | rgb | Vegetation index or indices to be calculated |
| `-mi`, `-inputs` | list-string | r,g,b | Model inputs (will be used in conjuction with `-index` flag options) |
| `-mn`, `-nodes` | list-integer | 8,8,8 | Number of nodes per model layer (by default specifies the number of layers) |
| `-md`, `-dropout` | float | 0.2 | Probability of model layer dropout (used to avoid overfitting) |
| `-mes`, `-earlystop` | list-integer,float | 5,0.001 | Early stop criteria ([patience],[change_threshold]) |
| `-te`, `-epochs` | integer | 100 | Number of training epochs (maximum number) |
| `-tb`, `-batch` | integer | 100 | Batch size |
| `-tc`, `-cache` | boolean | True | Cache batches (improves training time) |
| `-tp`, `-prefetch` | boolean | True | Prefetch batches (significantly improves training time) |
| `-tsh`, `-shuffle` | boolean | True | Shuffle inputs (use only for training to avoid overfitting) |
| `-tsp`, `-split` | float | 0.7 | Data split for model training (remainder will be used for model validation) |
| `-tci`, `-imbalance` | boolean | True | Adjust data inputs for class imbalance (will use lowest number of inputs) |
|`-tdr`, `-reduction` | float | 0.0 | Data reduction as proportion of 1.0 (useful if working with limited computing resources) |
| `-thresh`, `-threshold` | float | 0.6 | Confidence threshold used for reclassification |
| `-rad`, `-radius` | float | 0.10 | Radius used to compute geometry metrics (if specified in inputs) |

### Option A: Running as two separate programs
If utilizing the two program approach, first build, train, and save the model (line 1). Then, reclassify a LAS/LAZ file using one or more models (line 2):
```
python ML_veg_train.py
python ML_veg_reclass.py
```

#### ML_veg_train.py
##### Inputs:

The following inputs are required for the ML_veg_train program. If any of these options are not specified in the command line arguments, a pop-up window will appear for each.

1. The point cloud containing vegetation points only
2. The point cloud containing only bare-Earth points
3. The output model name

##### Outputs:
An output CSV file will be generated with the following naming scheme:

> {veg_filename}\_{noveg_filename}.csv

Where *{veg_filename}* is the file name of the point cloud containing vegetation points only and *{noveg_filename}* is the name of the point cloud containing bare-Earth points only.

The output CSV will have the following attributes (columns) of information:

> {vegetation_index_name}     {minimum_possible_index_value}      {maximum_possible_index_value}      {M-statistic}       {Otsu_threshold_value}

### Option B: Running as one single programs

#### veg_reclass.py
##### Inputs:

The program will automatically request the user to select 3 input files.

1. The point cloud to be reclassified.
2. The CSV file containing the vegetation index value ranges, M-statistics, and Otsu threshold values.
          --> This file is automatically created when you run veg_train.py

##### Outputs:

A new LAZ file will be generated with the following naming scheme:

> {filename}\_reclass\_{vegetation_index_name}\_veg\_noveg.las

Where *{filename}* is the original point cloud file name and *{vegetation_index_name}* is the name of the vegetation index determined or selected to differentiate vegetation from bare-Earth using Otsu's thresholding approach.

The output LAZ file will be saved in the same directory as the input file and will contain all the original points with updated classification values corresponding to either vegetation or bare-Earth.

# FEEDBACK
**If you have any questions about how to implement the code, suggestions for improvements, or feedback, please leave a comment or report the issue with as much detail as possible.**
