# TFM - Capacity Prediction Model
This repository contains my master's degree thesis in Big Data, Data Science and Artificial Intelligence.


## How to test
There is a copy of the .csv created from the extraction notebooks (sectors_dataset_2.csv), copy that into a folder and create a configuration.py file in /utils witht the following lines:

# path where datasets are saved
OUTPUT_PATH=
# datasets created with data extraction notebooks
SECTORS_DATASET_1=OUTPUT_PATH+'\\sectors_dataset_1.csv'
SECTORS_DATASET_2=OUTPUT_PATH+'\\sectors_dataset_2.csv'
# final dataset after EDA
DATASET_FINAL = OUTPUT_PATH + '\\sectors_dataset_final.csv'

Having the SECTORS_DATASET_2 path set correctly will be enough to run the EDA and following notebooks.

