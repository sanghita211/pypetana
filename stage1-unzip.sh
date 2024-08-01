#!/bin/bash

# set number of threads we want to use
THREADS=3

# path to the provided dir
cd $1

# use xargs to unzip all zip files (1.zip, 2.zip and 3.zip) found in the specified directory
find . -maxdepth 2 -iname '*.zip' | xargs -I {} -P $THREADS bash -c 'UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip "$1"' _ {}

# wait for files to unzip
wait
