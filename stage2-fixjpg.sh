#!/bin/bash

# limit Image Magick's internal threading
export MAGICK_THREAD_LIMIT=1

# set number of threads we want to use
THREADS=16

# path to the provided dir
cd $1

for DIR in 1 2 3
do
    if [ -d "$DIR" ]; then
        # path to dir
	cd $DIR

        # use xargs to launch all renames, and wait for all threads to finish before moving to next stage
        # (note that files on TUFTS are actually PBM formatted images, not JPEG)
	find . -maxdepth 2 -iname '*.jpg' | xargs -I {} -P $THREADS bash -c 'mv "$1" "${1%.*}.pbm"' _ {}
	wait

        # use xargs to launch all conversions from pbm to png; compresses images (lossless) to save space
	find . -maxdepth 2 -iname '*.pbm' | xargs -I {} -P $THREADS bash -c 'nice magick convert "$1" "${1%.*}.png"' _ {}
	wait

        # remove the old pbm, which are 3x the size of the pngs and serve no purpose now
	find . -maxdepth 2 -iname '*.pbm' | xargs -I {} -P $THREADS bash -c 'rm "$1"' _ {}
	wait

        # path back for the next scanner directory
	cd ..
    fi
done
