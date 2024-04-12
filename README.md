# Programs for Segmenting Vegetation from bare-Earth in High-relief and Dense Point Clouds

These programs are designed to segment vegetation from bare-Earth points in a dense point cloud, although they may also be used to segment any two classes that are visually distinguishable from each other. The programs are meant to reclassify large and dense point clouds, similar to the following:

<img src='/misc/images/FIGURE_20200508_RGB.png' alt='R G B color model of a coastal bluff near Port Angeles, WA'>

And reclassify the points, similar to the following:

<img src='/misc/images/FIGURE_20200508_RGB_16.png' alt='model of a coastal bluff colored by classification'>

Green points represent 'vegetation' and brown points represent 'bare-Earth'.

There are two approaches:

1. The [non-machine learning approach](#usage-programs-without-machine-learning) utilizes a vegetation index and Otsu's thresholding method. This approach is more computationally demanding and the appropriate vegetation index and threshold value are likely to vary by location and application.
2. The [machine learning approach](#usage-for-machine-learning-programs) utilizes the Tensorflow API. This approach is more efficient, less subjective, and more robust across geographies and applications. Although it is faster with access to a GPU, a CPU can also be used. *(Note: The code is not optimized for use with multiple GPUs.)*

The full paper describing this approach is currently in review by *Remote Sensing*:
Wernette, Phillipe A. (in reivew) Vegetation Filtering of Coastal Cliff and Bluff Point Clouds with MLP. Submitted to *Remote Sensing*.

# Contents

1. [Vegetation Indices Included](#vegetation-indices-ncluded)
2. [Installation](#installation)
3. [Non-Machine Learning Approach](#usage-programs-without-machine-learning)
4. [Machine Learning Approach](#usage-for-machine-learning-programs)
   - [Command Line Arguments](#command-line-arguments)
   - [Option A: Two Separate Programs](#option-a-run-two-separate-programs)
     - [ML_veg_train.py](#ml_veg_trainpy)
     - [ML_veg_reclass.py](#ml_veg_reclasspy)
   - [Option B: One Single Program](#option-b-run-a-single-program)
     - [ML_vegfilter.py](#ml_vegfilterpy)
5. [Feedback](#feedback)
6. [References](#references)

## Vegetation Indices Included

These programs compute the following vegetation indicies, their M-statistics, and Otsu threshold values:

- Excess Red (exr)[^1]
- Excess Green (exg)[^2]
- Excess Green-Red (exgr)[^3]
- Excess Blue (exb)[^4]
- Normal Green-Red Difference Index (ngrdi)[^5]
- Modified Green Red Vegetation Index (mgrvi)[^5]
- Green Leaf Index (gli)[^6]
- Red Green Blue Veggetation Index (rgbvi)[^7]
- Kawashima Index (ikaw)[^8]
- Green Lead Algorithm (gla)[^6]

Each of the vegetation indices only requires some combination of the red, green, and blue color bands. No NIR, SWIR, or other band is required.

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

# Installation:

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

# USAGE (PROGRAMS *WITHOUT* MACHINE LEARNING):

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

## veg_train.py
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

## veg_reclass.py
### Inputs:

The program will automatically request the user to select 2 input files.

1. The point cloud to be reclassified.
2. The CSV file containing the vegetation index value ranges, M-statistics, and Otsu threshold values.
          --> This file is automatically created when you run veg_train.py

### Outputs:

A new LAZ file will be generated with the following naming scheme:

> {filename}\_reclass\_{vegetation_index_name}\_veg\_noveg.laz

Where *{filename}* is the original point cloud file name and *{vegetation_index_name}* is the name of the vegetation index determined or selected to differentiate vegetation from bare-Earth using Otsu's thresholding approach.

The output LAZ file will be saved in the same directory as the input file and will contain all the original points with updated classification values corresponding to either vegetation or bare-Earth.


# USAGE (FOR MACHINE LEARNING PROGRAMS):

The machine learning approach can be run [(1) as two separate programs](#option-a-run-two-separate-programs), one for ML model training and a second for LAS/LAZ file (re)classification, or [(2) as a single program](#option-b-run-a-single-program) that builds and trains a ML model and then uses that model to reclassify a LAS/LAZ file.

### Command Line Arguments

Command line options are available to for both the two program and one program options to cut down on pop-up windows and aid in batch scripting:
| Argument | Type(s) | Default value(s) | Description/Function | Program |
| --- | --- | --- | --- | --- |
| `-v`, `-veg` | string | NA | Point cloud containing vegetation points only | ML_veg_train, ML_vegfilter |
| `-g`, `-ground` | string | NA | Point cloud containing ground points only | ML_veg_train, ML_vegfilter |
| `-r`, `-reclass` | string | NA | Point cloud to be reclassified | ML_veg_reclass, ML_vegfilter |
| `-h5`, `-model` | string | NA | h5 Model file | ML_veg_reclass, ML_vegfilter |
| `-m`, `-name` | string | NA | ML model name | ML_veg_train, ML_vegfilter |
| `-vi`, `-index` | string | rgb | Vegetation index or indices to be calculated | ML_veg_train, ML_vegfilter |
| `-mi`, `-inputs` | list-string | r,g,b | Model inputs (will be used in conjuction with `-index` flag options) | ML_veg_train, ML_vegfilter |
| `-mn`, `-nodes` | list-integer | 8,8,8 | Number of nodes per model layer (by default specifies the number of layers) | ML_veg_train, ML_vegfilter |
| `-md`, `-dropout` | float | 0.2 | Probability of model layer dropout (used to avoid overfitting) | ML_veg_train, ML_vegfilter |
| `-mes`, `-earlystop` | list-integer,float | 5,0.001 | Early stop criteria ([patience],[change_threshold]) | ML_veg_train, ML_vegfilter |
| `-te`, `-epochs` | integer | 100 | Number of training epochs (maximum number) | ML_veg_train, ML_vegfilter |
| `-tb`, `-batch` | integer | 100 | Batch size | ML_veg_train, ML_veg_reclass, ML_vegfilter |
| `-tc`, `-cache` | boolean | True | Cache batches (improves training time) | ML_veg_train, ML_veg_reclass, ML_vegfilter |
| `-tp`, `-prefetch` | boolean | True | Prefetch batches (significantly improves training time) | ML_veg_train, ML_veg_reclass, ML_vegfilter |
| `-tsh`, `-shuffle` | boolean | True | Shuffle inputs (use only for training to avoid overfitting) | ML_veg_train, ML_veg_reclass, ML_vegfilter |
| `-tsp`, `-split` | float | 0.7 | Data split for model training (remainder will be used for model validation) | ML_veg_train, ML_vegfilter |
| `-tci`, `-imbalance` | boolean | True | Adjust data inputs for class imbalance (will use lowest number of inputs) | ML_veg_train, ML_vegfilter |
|`-tdr`, `-reduction` | float | 0.0 | Data reduction as proportion of 1.0 (useful if working with limited computing resources) | ML_veg_train, ML_vegfilter |
| `-thresh`, `-threshold` | float | 0.6 | Confidence threshold used for reclassification | ML_veg_reclass, ML_vegfilter |
| `-rad`, `-radius` | float | 0.10 | Radius used to compute geometry metrics (if specified in inputs) | ML_veg_train, ML_veg_reclass, ML_vegfilter |


## OPTION A: RUN TWO SEPARATE PROGRAMS
If utilizing the two program approach, first build, train, and save the model (line 1). Then, reclassify a LAS/LAZ file using one or more models (line 2):
```
python ML_veg_train.py
python ML_veg_reclass.py
```

## ML_veg_train.py
### Inputs:

The following inputs are required for the ML_veg_train program. If any of these options are not specified in the command line arguments, a pop-up window will appear for each.

1. The point cloud containing vegetation points only
2. The point cloud containing only bare-Earth points
3. The output model name

The ML_veg_train program will read in the two training point clouds, account for any class imbalance, build a ML model, and train the ML model.

### Outputs:
All outputs will be saved in a directory with the following scheme:

> saved_models_{date}

Where *{date}* is the date the model was created and is pulled from the computer clock. If this directory does not already exist then it will first be created.

The trained model will be written out as a single h5 file as well as a directory. Both the h5 file and the directory will have the same name, as specified by the user.

A plot of the model will also be saved as a PNG file (see example below), and a summary text file will be written that contains the complete model summary and metadata.

<img src='/misc/images/rgb_8_GRAPH.png' alt='R G B model with one layer of 8 nodes' height=50% width=50%>

## ML_veg_reclass.py
### Inputs:

The following inputs are required for the ML_veg_reclass program. If any of these options are not specified in the command line arguments, a pop-up window will appear for each.

1. The point cloud to be reclassified
2. The h5 model file (can be generated using the ML_veg_train program)

The ML_veg_reclass program will automatically read in the model structure, weights, and required inputs (including vegetation indices and geometry metrics) and will reclassify the input point cloud.

### Outputs:
The reclassified LAS/LAZ file will be saved in a directory with the following scheme:

> results_{date}

Where {date} is the date the model was created and is pulled from the computer clock. If this directory does not already exist then it will first be created.

A new LAZ file will be generated in the results directory:

> {filename}_{model_name}_{threshold_value}.laz

Where *{filename}* is the original point cloud file name, *{model_name}* is the name of the model used to reclassify the input point cloud, and *{threshold_value}* is the threshold value used to segment vegetation from bare-Earth.


## OPTION B: RUN A SINGLE PROGRAM

## ML_vegfilter.py

The ML_vegfilter program will use the two training point clouds to generate a machine learning model with the user-specified arguments, and then use this model to reclassify the specified point cloud. The significant advantage of using a single program is eliminating the need to read the model file for reclassification.

### Inputs:

The following inputs are required for the ML_vegfilter program. If any of these options are not specified in the command line arguments, a pop-up window will appear for each.

1. The point cloud containing vegetation points only
2. The point cloud containing only bare-Earth points
3. The output model name
4. The point cloud to be reclassified

### Outputs:

The model will be saved as a h5 file and a directory, as well as a PNG of the model structure and a detailed metadata summary text file. The model and all it's associated files (graph as PNG and summary metadata file) will be saved in a *saved_models_{date}* folder, where {date} is the date the model was created.

The reclassified point cloud will be saved in the *results_{date}* folder as:

> {filename}_{model_name}_{threshold_value}.laz

Where *{filename}* is the original point cloud file name, *{model_name}* is the name of the model used to reclassify the input point cloud, and *{threshold_value}* is the threshold value used to segment vegetation from bare-Earth.

# FEEDBACK

**If you have any questions about how to implement the code, suggestions for improvements, or feedback, please leave a comment or report the issue with as much detail as possible.**

# REFERENCES
[^1]: Meyer, G.E.; Neto, J.C. 2008. Verification of color vegetation indices for automated crop imaging applications. Comput. Electron. Agric. 63, 282–293.
[^2]: Woebbecke, D.M.; Meyer, G.E.; Von Bargen, K.; Mortensen, D.A. 1995. Color Indices forWeed Identification Under Various Soil, Residue, and Lighting Conditions. Trans. ASAE, 38, 259–269.
[^3]: Mao, W.; Wang, Y.;Wang, Y. 2003. Real-time detection of between-row weeds using machine vision. In Proceedings of the 2003 ASAE Annual Meeting; American Society of Agricultural and Biological Engineers, Las Vegas, NV, USA, 27–30 July 2003.
[^4]: Neto, J.C. 2004. A combined statistical-soft computing approach for classification and mapping weed species in minimum -tillage systems. Ph.D. Thesis, University of Nebraska – Lincoln, Lincoln, NE, USA, August 2004.
[^5]: Tucker, C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens. Environ. 8, 127–150.
[^6]: Louhaichi, M.; Borman, M.M.; Johnson, D.E. 2001. Spatially located platform and aerial photography for documentation of grazing impacts on wheat. Geocarto Int. 16, 65–70.
[^7]: Bendig, J.; Yu, K.; Aasen, H.; Bolten, A.; Bennertz, S.; Broscheit, J.; Gnyp, M.L.; Bareth, G. 2015. Combining UAV-based plant height from crop surface models, visible, and near infrared vegetation indices for biomass monitoring in barley. Int. J. Appl. Earth Obs. Geoinf. 39, 79–87.
[^8]: Kawashima, S.; Nakatani, M. 1998. An algorithm for estimating chlorophyll content in leaves using a video camera. Ann. Bot. 81, 49–54.
