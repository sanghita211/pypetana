#!/usr/bin/env python3
"""stage3-preprocess.py
Description:
  Targets a series of Physarum images and extracts all petri dishes, masking weights
  and writing a new series of images for each dish. Additionally rotates individual
  petri dishes which have masses, using the masses position and count to identify
  angle of rotation.

Usage:
    stage3-preprocess.py [options] <patterns>...

    stage3-preprocess.py -h | --help
Options:
  -h, --help                    Show this screen.
"""

import os
import glob
import cv2
import numpy as np
from docopt import docopt

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def apply_curve_transformation(image, transformation_function, lower_threshold, upper_threshold):
    # Ensure image is in a float format to avoid clipping during the transformation
    float_image = image.astype(np.float32) / 255.0

    # Apply the transformation function
    transformed_image = transformation_function(float_image, lower_threshold, upper_threshold)

    # Clip values to the 0-1 range and convert back to an 8-bit format
    transformed_image = np.clip(transformed_image, 0, 1) * 255
    transformed_image = transformed_image.astype(np.uint8)

    return transformed_image

# Example of a simple linear transformation function
def linear_transformation(pixel_values):
    return 1 - pixel_values  # Simple inversion for demonstration

def piecewise_linear_transformation(pixel_values_in, lower_threshold, upper_threshold):
    invert=True

    # invert it
    pixel_values = 1 - pixel_values_in if invert else pixel_values_in

    # Clip values at the lower and upper thresholds
    clipped_values = np.clip(pixel_values, lower_threshold, upper_threshold)
    
    # Linear scaling
    # Scale the range [lower_threshold, upper_threshold] to [0, 1]
    scale = 1.0 / (upper_threshold - lower_threshold)
    transformed = (clipped_values - lower_threshold) * scale
    
    # invert it back
    transformed = 1 - transformed if invert else transformed

    return transformed

def eliminate_duplicate_circles(circles, center_threshold):
    unique_indices = []

    for index, circle in enumerate(circles):
        x, y, r = circle
        duplicate_found = False

        for index2 in unique_indices:
            ux, uy, ur = circles[index2]
            distance = np.sqrt((ux - x)**2 + (uy - y)**2)

            if distance < center_threshold:
                duplicate_found = True
                break

        if not duplicate_found:
            unique_indices.append(index)

    return np.array(unique_indices,dtype=int)

def extract_radius_mask(filepath, save=None):
    frame = cv2.imread(filepath)
    modified_frame = apply_curve_transformation(frame, piecewise_linear_transformation, 0.00, 0.70)
    gray = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    dish_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=300,
                               param1=20, param2=29, minRadius=1150, maxRadius=1350)
    unique_indices = eliminate_duplicate_circles(dish_circles[0], center_threshold=300)
    dish_circles = np.round(dish_circles[0, unique_indices]).astype("int")

    # sort the dish circles left to right first, then top to bottom
    height, width, channels = frame.shape
    left_dish_circles = dish_circles[np.where(dish_circles[:,0] < 6*width/10)]
    right_dish_circles = dish_circles[np.where(dish_circles[:,0] >= 6*width/10)]
    left_dish_circles = left_dish_circles[np.argsort(left_dish_circles[:,1])]
    right_dish_circles = right_dish_circles[np.argsort(right_dish_circles[:,1])]
    dish_circles = np.concatenate((left_dish_circles, right_dish_circles), axis=0)

    # Draw the filled dish circle on the mask as white (keeping)
    mask = np.zeros_like(frame)
    for (x, y, r) in dish_circles:
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Mask the image before searching for the weights
    masked_frame = cv2.bitwise_and(modified_frame, mask)
    blurred = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    weight_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT_ALT, dp=1, minDist=1,
                               param1=60, param2=0.30, minRadius=90, maxRadius=140)[0]

    # sort by radius here
    radii_indices = np.argsort(weight_circles[:,2])[::-1]
    weight_circles = weight_circles[radii_indices]

    unique_indices = eliminate_duplicate_circles(weight_circles, center_threshold=180)
    weight_circles = np.round(weight_circles[unique_indices]).astype("int")


    # Draw the filled weight circle on the mask as black (discard)
    for (x, y, r) in weight_circles:
        in_center = False
        for (x2, y2, r2) in dish_circles:
            if np.sqrt((x-x2)**2+(y-y2)**2) < 800:
                in_center = True
                break
        if in_center == False:
            cv2.circle(mask, (x, y), r+20, (0, 0, 0), -1)


    # Now for each dish we calculate necessary rotation to place the
    # three masses on the right (centered at 0 radians) and the single
    # mass on the left (centered at pi radians)
    rot_angles = []
    for i, (x1, y1, r2) in enumerate(dish_circles):
        left_weights_x = []
        left_weights_y = []
        right_weights_x = []
        right_weights_y = []
        for j, (x2, y2, r2) in enumerate(weight_circles):
            if np.sqrt((x1-x2)**2+(y1-y2)**2) < 800:
                continue
            if (x1-x2)**2+(y1-y2)**2 <= 1250*1250:
                if x2 < x1:
                    left_weights_x.append(x2)
                    left_weights_y.append(y2)
                else:
                    right_weights_x.append(x2)
                    right_weights_y.append(y2)
        if len(left_weights_x) > 0 and len(right_weights_x) > 0:
            xl, yl = np.average(left_weights_x), np.average(left_weights_y)
            xr, yr = np.average(right_weights_x), np.average(right_weights_y)
            theta = np.arctan2(yr-y1,xr-x1)
            if len(left_weights_x) > len(right_weights_x):
                theta += np.pi
            rot_angles.append(theta)
        else:
            rot_angles.append(0)

    # optionally output the masked first frame
    if save is not None:
        final_image = np.zeros((height, width * 2, channels), dtype=np.uint8)

        final_image[0:height, 0:width] = frame
        masked_frame = cv2.bitwise_and(frame, mask)
        final_image[0:height, width:2*width] = masked_frame
        for i, (x, y, r) in enumerate(dish_circles):
            # extract the single dish
            dish = masked_frame[y - r:y + r, x - r:x + r]
            # rotate it
            rotation_matrix = cv2.getRotationMatrix2D((int(r),int(r)), rot_angles[i]*180/np.pi, 1)
            rotated_dish = cv2.warpAffine(dish, rotation_matrix, (2*int(r), 2*int(r)))
            # re-insert it
            final_image[y - r:y + r, x - r + width:x + r + width] = rotated_dish
            cv2.putText(final_image, f'd{i+1:02}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
            cv2.putText(final_image, f'r{rot_angles[i]*180/np.pi:5.2}', (x, y+100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
        cv2.imwrite(save, final_image)

    return mask, dish_circles, rot_angles

def process_series(series_prefix, series, min_radius, min_count):
    date, scanner, experiment, files, mask, dish_coords, dish_angles, first_frame, last_frame = series
    for i, (x, y, r) in enumerate(dish_coords): print(f'output/{series_prefix}_{i+1:02}_d03.jpg')

    if len(dish_coords) < 6:
        print(f'WARNING: PREPROCESSED/{series_prefix}_%d_%03d has {len(dish_coords)} dishes, expected 6')

    # get the first good image
    last_frame = {}
    for filepath in files:
        try:
            frame = cv2.imread(filepath)
            masked_frame = cv2.bitwise_and(frame, mask)
            for i, (x, y, r) in enumerate(dish_coords):
                # Ensure the circle is fully within the frame
                x, y, r = max(x, min_radius), max(y, min_radius), min(min_radius, min(x, y, masked_frame.shape[1] - x, masked_frame.shape[0] - y))
                
                # Extract the Petri dish
                dish = masked_frame[y - min_radius:y + min_radius, x - min_radius:x + min_radius]
                dish = apply_curve_transformation(dish, piecewise_linear_transformation, 0.0, 0.85)
                last_frame[i] = dish
        except:
            continue

    this_frame_index = 1
    frame_index = 0
    for filepath in files:
        frame = cv2.imread(filepath)
        frame_index = int(filepath.split('_')[-1].split('.')[0])
        if frame is None:
            continue

        while this_frame_index < frame_index:
            for i, (x, y, r) in enumerate(dish_coords):
                this_last_frame = last_frame[i].copy()
                cv2.putText(this_last_frame, f'f{this_frame_index:03}', (105, 105), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                cv2.circle(this_last_frame, (50, 50), 40, (50, 50, 250), -1)
                cv2.imwrite(f'PREPROCESSED/{series_prefix}_{i+1:02}_{this_frame_index:03}.jpg', this_last_frame)
            this_frame_index += 1

        if dish_coords is not None and mask is not None:
            masked_frame = cv2.bitwise_and(frame, mask)
            for i, (x, y, r) in enumerate(dish_coords):
                # Ensure the circle is fully within the frame
                x, y, r = max(x, min_radius), max(y, min_radius), min(min_radius, min(x, y, masked_frame.shape[1] - x, masked_frame.shape[0] - y))
                
                # Extract the Petri dish
                dish = masked_frame[y - min_radius:y + min_radius, x - min_radius:x + min_radius]
                
                # Rotate it
                if dish_angles[i] != 0:
                    rotation_matrix = cv2.getRotationMatrix2D((int(min_radius),int(min_radius)), dish_angles[i]*180/np.pi, 1)
                    dish = cv2.warpAffine(dish, rotation_matrix, (2*int(min_radius), 2*int(min_radius)))

                # Check if dish is not empty
                if dish.size > 0 and dish.shape[0] > 0 and dish.shape[1] > 0:
                    # do a levels adjustment, per color channel
                    dish = apply_curve_transformation(dish, piecewise_linear_transformation, 0.0, 0.85)
                    last_frame[i] = dish.copy()

                    # Put a green dot in the top right corner
                    cv2.circle(dish, (50, 50), 40, (50, 250, 50), -1)

                    # Add some text to mark the source filename and dish
                    cv2.putText(dish, f'f{frame_index:03}', (105, 105), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

                    cv2.imwrite(f'PREPROCESSED/{series_prefix}_{i+1:02}_{frame_index:03}.jpg', dish)
        this_frame_index += 1

# Get the number of series
if __name__ == '__main__':
    args = docopt(__doc__)
    patterns = args['<patterns>']

    ensure_directory_exists('PREPROCESSED')

    min_radius = 0
    min_count = 0
    
    series_dict = {}
    for date_dir in sorted(next(os.walk('.'))[1]):
        if 'git' not in date_dir and 'PROCESSED' not in date_dir:
            print(date_dir)
            for scanner_dir in sorted(next(os.walk(date_dir))[1]):
                pattern = date_dir + '/' + scanner_dir + '/*'
                files = sorted(glob.glob(pattern))
                if len(files) > 0:
                    experiment_number = files[0].split('.')[0].split('/')[-1]
                    key = date_dir + '_' + scanner_dir + '_' + experiment_number
                    if sum([p in key for p in patterns]):
                        first_frame = int(files[0].split('_')[-1].split('.')[0])
                        last_frame = int(files[-1].split('_')[-1].split('.')[0])
                        mask, dish_circles, rot_angles = extract_radius_mask(files[0], save=f'PREPROCESSED/preprocess_{key}.jpg')
                        series_dict[key] = [
                                date_dir,
                                scanner_dir,
                                experiment_number,
                                files,
                                mask,
                                dish_circles,
                                rot_angles,
                                first_frame,
                                last_frame
                        ]
                        if max(dish_circles[:,2]) > min_radius:
                            min_radius = max(dish_circles[:,2])
                        if len(files) > min_count:
                            min_count = len(files)

    for key, series in series_dict.items():
        process_series(key, series, min_radius, min_count)
