#!/bin/bash

for scanner in 1 2 3
do
    for dish in 01 02 03 04 05 06
    do
        if [ -d "PROCESSED/nh_fracdim_$1_$scanner_$2_$dish_001.jpg" ]; then
            ffmpeg -framerate 5 -i PROCESSED/nh_fracdim_$1_$scanner_$2_$dish_%03d.jpg -c:v libx264 -r 5 -pix_fmt yuv420p PROCESSED/nh_fracdim_$1_$scanner_$2_$dish.mp4
            ffmpeg -framerate 5 -i PROCESSED/nh_physarum_$1_$scanner_$2_$dish_%03d.jpg -c:v libx264 -r 5 -pix_fmt yuv420p PROCESSED/nh_physarum_$1_$scanner_$2_$dish.mp4
            ffmpeg -framerate 5 -i PROCESSED/nh_fracdim_$1_$scanner_$2_$dish_%03d_plt.jpg -c:v libx264 -r 5 -pix_fmt yuv420p PROCESSED/nh_fracdim_$1_$scanner_$2_$dish_plt.mp4
        fi
    done
done
