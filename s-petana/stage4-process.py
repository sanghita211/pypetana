#!/usr/bin/env python3
"""circularity.py
Description:
    Calculates the perimeter and area occupied by physarum through a series of
    frames.

Usage:
    circularity.py [--min_thresh <val> --max_thresh <val> --side=<val> --include_holes] <filenames>...

    curcularity.py -h | --help Options:
  -h, --help                    Show this screen.  --side=<val>                  Side to process [default: both]
  --min_thresh=<val>            Integer between 0 and 255, inclusive [default: 20]
  --max_thresh=<val>            Integer between 0 and 255, inclusive [default: 164]
  --include_holes               Flag to add perimeter and subtract area of holes in all contours
"""

import os
import glob
import cv2
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_contour_inside(contour1, contour2):
    # Check if all points of contour1 are inside contour2
    for i in range(len(contour1)):
        # Using cv2.pointPolygonTest to check each point of contour1 against contour2
        point = (int(contour1[i][0][0]), int(contour1[i][0][1]))
        if cv2.pointPolygonTest(contour2, point, False) < 0:
            return False
    return True

def generate_points(last_point, point, N):
    """Generate N points between last_point and point."""
    x1, y1 = last_point
    x2, y2 = point

    points = []
    for i in range(1, N + 1):
        t = i / (N + 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((int(round(x)), int(round(y))))

    return points

def is_contour_partially_inside(contour1, contour2, side_width):
    # Check if all points of contour1 are inside contour2
    last_point = (int(contour1[-1][0][0]), int(contour1[-1][0][1]))
    for i in range(len(contour1)):
        # Using cv2.pointPolygonTest to check each point of contour1 against contour2
        point = (int(contour1[i][0][0]), int(contour1[i][0][1]))
        if cv2.pointPolygonTest(contour2, point, False) > 0:
            return True
        points = generate_points(last_point, point,int(side_width/2)+1)
        for new_point in points:
            if cv2.pointPolygonTest(contour2, new_point, False) > 0:
                return True
        last_point = point
    return False

def is_contour_intersecting(contour1, contour2, side_width):
    # Check if all points of contour1 are inside contour2
    last_point = (int(contour1[-1][0][0]), int(contour1[-1][0][1]))
    for i in range(len(contour1)):
        # Using cv2.pointPolygonTest to check each point of contour1 against contour2
        point = (int(contour1[i][0][0]), int(contour1[i][0][1]))
        if cv2.pointPolygonTest(contour2, point, False) == 0:
            return True
        points = generate_points(last_point, point,side_width)
        for new_point in points:
            if cv2.pointPolygonTest(contour2, new_point, False) == 0:
                return True
        last_point = point
    return False

def points_in_contour(points, contour1):
    # gather any points in the contour
    new_points = []
    for i in range(len(points)):
        # Using cv2.pointPolygonTest to check each point of contour1 against contour2
        point = points[i]
        if cv2.pointPolygonTest(contour1, point, False) >= 0:
            new_points.append(point)
    return new_points

def find_and_sort_contours(gray_image, min_thresh, max_thresh, last_center):
    # Apply range threshold
    range_threshold_image = cv2.inRange(gray_image, min_thresh, max_thresh)
    range_threshold_image = cv2.medianBlur(range_threshold_image, 9)

    # Fill in holes
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the closing operation
    range_threshold_image = cv2.morphologyEx(range_threshold_image, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(range_threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # sort contours
    grouped_contours = []
    weights = []
    index = 0
    while index < len(sorted_contours):
        contour = sorted_contours[index]

        contour_list = []
        contour_list.append(contour)

        # calculate the center of the contour
        M = cv2.moments(contour)
        # Calculate the center (centroid) of the contour
        if M["m00"] != 0:
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
        else:
            centerX, centerY = 0, 0  # Set default values if m00 is zero to avoid division by zero


        # Calculate area
        area = cv2.contourArea(contour)

        # Calculate perimeter
        if area > 0:
            perimeter = cv2.arcLength(contour, True)  # True indicates the contour is closed

            index_next = index + 1
            while index_next < len(sorted_contours):
                contour_next = sorted_contours[index_next]
                area2 = cv2.contourArea(contour_next)
                if area2 > 0:
                    perimeter2 = cv2.arcLength(contour_next, True)  # True indicates the contour is closed
                    if is_contour_inside(contour_next, contour):
                        if args['--include_holes']:
                            perimeter += perimeter2  # True indicates the contour is closed
                            area -= area2
                            contour_list.append(contour_next)
                        sorted_contours.pop(index_next)
                    else:
                        index_next += 1
                else:
                    sorted_contours.pop(index_next)

            grouped_contours.append([np.array([centerX, centerY]), area, perimeter, contour_list])
            
            # calculate distance from center/area, g
            weights.append(np.sqrt((centerX-last_center[0])**2 + (centerY-last_center[1])**2)/area)
            index += 1
        else:
            sorted_contours.pop(index)
    
    sorted_indices = np.argsort(np.array(weights))
    last_center = grouped_contours[sorted_indices[0]][0]

    return sorted_contours, grouped_contours, sorted_indices, last_center, weights

# Get the number of series
if __name__ == '__main__':
    args = docopt(__doc__)
    image_paths = sorted(args['<filenames>'])
    min_thresh = int(args['--min_thresh'])
    max_thresh = int(args['--max_thresh'])
    side = args['--side']

    ensure_directory_exists('PROCESSED')

    fileout_filename = 'nh_fracdim_'+'_'.join(image_paths[0].split('/')[-1].replace('.jpg','').split('_')[:-1])+'.dat'
    if side in ['left','right']:
        fileout_filename = side + '_' + fileout_filename

    fileout_filename_frac = 'nh_fracdim_'+'_'.join(image_paths[0].split('/')[-1].replace('.jpg','').split('_')[:-1])+'_frac.dat'
    with open('PROCESSED' + '/' + fileout_filename, 'w') as fileout, open('PROCESSED' + '/' + fileout_filename_frac, 'w') as fileout_frac:
        output = (f'#{"frame":>8}  {"area":>15}  {"perimeter":>15}  {"circularity":>15}  {"fracdim":>15}\r\n')
        output_frac = (f'#{"frame":>8}  {"BSR":>15}  {"NA":>15}  {"NP":>15}\r\n')

        last_center = None
        check_split = None
        for image_index, image_path in enumerate(image_paths):
            # Load the image
            image = cv2.imread(image_path)
            physarum_only_image = cv2.imread(image_path)
            frame = int(image_path.split('.')[-2].split('_')[-1])

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            channels = 3

            if last_center is None:
                last_center = (int(width/2), int(height/2))
            
            # Apply a median filter to reduce noise and increase connectivity
            gray_image = cv2.medianBlur(gray_image, 9)

            # Generate and apply mask to remove edges
            mask = np.zeros_like(gray_image)
            cv2.circle(mask, (int(width/2), int(height/2)), int(0.465*width), (255), -1)
            gray_image = cv2.bitwise_and(gray_image, mask)

            if image_index == 0:
                first_sorted_contours, first_grouped_contours, first_sorted_indices, first_center, _= find_and_sort_contours(gray_image, min_thresh, max_thresh, last_center)
            if side in ['right','left']:
                first_split_y = [i for i in range(100,int(height-100))]
                first_split = []
                for i, val in enumerate(first_split_y):
                    if side == 'right':
                        first_split.append([int(first_center[0]), val])
                    else:
                        first_split.append([int(first_center[0]), val])
                #first_split = points_in_contour(first_split, first_grouped_contours[first_sorted_indices[0]][3][0])

            # Generate an additional mask to remove the left or right side of the image, optionally
            if side == 'right':
                mask = np.zeros_like(gray_image)
                cv2.rectangle(mask, (int(first_center[0]), 0), (int(width), int(height)), (255), -1)
                gray_image = cv2.bitwise_and(gray_image, mask)
            if side == 'left':
                mask = np.zeros_like(gray_image)
                cv2.rectangle(mask, (0, 0), (int(first_center[0]), int(height)), (255), -1)
                gray_image = cv2.bitwise_and(gray_image, mask)

            sorted_contours, grouped_contours, sorted_indices, last_center, weights = find_and_sort_contours(gray_image, min_thresh, max_thresh, last_center)
            cv2.drawContours(image, sorted_contours, -1, (255, 0, 0), 3)

            # create four quadrants
            top_left_contour = np.array([
                [0, 0],
                [width/2, 0],
                [width/2, height/2],
                [0, height/2]
            ])
            top_left_contour = top_left_contour.reshape((-1, 1, 2))
            top_right_contour = np.array([
                [width/2, 0],
                [width, 0],
                [width, height/2],
                [width/2, height/2]
            ])
            top_right_contour = top_right_contour.reshape((-1, 1, 2))
            bot_left_contour = np.array([
                [0, height/2],
                [width/2, height/2],
                [width/2, height],
                [0, height]
            ])
            bot_left_contour = bot_left_contour.reshape((-1, 1, 2))
            bot_right_contour = np.array([
                [width/2, height/2],
                [width, height/2],
                [width, height],
                [width/2, height]
            ])
            bot_right_contour = bot_right_contour.reshape((-1, 1, 2))

            # Define the box size
            box_sizes = list([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
            points = []
            total_area = 0
            total_perimeter = 0
            logN = []
            logbsr = []

            # Create a mask with the same dimensions as the image
            physarum_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            for index in sorted_indices[:1]:
                center, area, perimeter, contours_list = grouped_contours[index]
                simplified_points = np.array([point[0] for point in cv2.approxPolyDP(contours_list[0], 10, True)])

                include_group = False
                if len(points) == 0:
                    include_group = True
                else:
                    for point in points:
                        for new_point in simplified_points:
                            if np.sqrt((point[0]-new_point[0])**2+(point[1]-new_point[1])**2) < 100:
                                include_group = True
                                break

                # Draw the contour on the mask and fill it
                cv2.drawContours(physarum_mask, [contours_list[0]], -1, 255, thickness=cv2.FILLED)

                if include_group:
                    # draw this contour and all related
                    cv2.drawContours(image, contours_list, -1, (0, 0, 255), 3)
                    # add the area and perimeter
                    total_area += area
                    total_perimeter += perimeter
                    if len(points) == 0:
                        points = simplified_points
                    else:
                        points = np.append(points, simplified_points, axis=0)

                    # remove the center line from the perimeter when doing left/right
                    if side in ['left','right']:
                        # test what points of the center line are inside the contour
                        first_split = points_in_contour(first_split, contours_list[0])

                        # every pixel is a coordinate, and the list of every pixel is 
                        # an effective perimeter portion we can remove
                        total_perimeter -= len(first_split)

                        # draw the white line for what was removed
                        for i in range(len(first_split) - 1):
                            start_point = first_split[i]
                            end_point = first_split[i + 1]

                            # check for gaps: only draw segments with no gaps
                            if abs(start_point[1]-end_point[1]) == 1:
                                cv2.line(image, start_point, end_point, (127, 127, 127), 3)

                    for bsi, box_size in enumerate(box_sizes):
                        # Loop over the image and create boxes
                        box_contours = []
                        for y in range(0, height, box_size):
                            for x in range(0, width, box_size):
                                # Define the box coordinates
                                if x + box_size < width and y + box_size < height:
                                    contour = np.array([
                                        [x, y],
                                        [x + box_size, y],
                                        [x + box_size, y + box_size],
                                        [x, y + box_size]
                                    ])
                                    contour = contour.reshape((-1, 1, 2))
                                    box_contours.append(contour)

                        box_contours_inside = []
                        for contour in box_contours:
                            for contour2 in contours_list:
                                if is_contour_partially_inside(contour, contour2, box_size):
                                    box_contours_inside.append(contour)
                        box_contours_intersecting = []
                        for contour in box_contours_inside:
                            if is_contour_intersecting(contour, contour2, box_size):
                                box_contours_intersecting.append(contour)
                        # Draw the contour
                        if box_size == 10:
                            for contour in box_contours_inside:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x < width/2 + 10 and avg_y < height/2 + 30:
                                    cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)
                            for contour in box_contours_intersecting:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x < width/2 + 10 and avg_y < height/2 + 30:
                                    cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)
                        if box_size == 20:
                            for contour in box_contours_inside:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x > width/2 and avg_y < height/2 + 20:
                                    cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)
                            for contour in box_contours_intersecting:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x > width/2 and avg_y < height/2 + 20:
                                    cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)
                        if box_size == 40:
                            for contour in box_contours_inside:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x > width/2 +20 and avg_y > height/2 +20:
                                    cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)
                            for contour in box_contours_intersecting:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x > width/2 +20 and avg_y > height/2 +20:
                                    cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)
                        if box_size == 80:
                            for contour in box_contours_inside:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x < width/2 + 20 and avg_y > height/2:
                                    cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)
                            for contour in box_contours_intersecting:
                                avg_x = np.average(contour[:,0,0])
                                avg_y = np.average(contour[:,0,1])
                                if avg_x < width/2 + 20 and avg_y > height/2:
                                    cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)

                        # calculate log(N) and log(1/bs)
                        logN.append(np.log(len(box_contours_inside)))
                        logbsr.append(np.log(box_size))
                        output_frac += f'{frame:>9} {box_size:16e} {len(box_contours_inside):16e} {len(box_contours_intersecting):16e}\r\n'
                    fractal_dimension, intercept = np.polyfit(logbsr, logN, 1)

            
            _, _, _, contours_list = grouped_contours[sorted_indices[0]]
            circularity = 4*np.pi*total_area/total_perimeter**2
            cv2.putText(image, f'{circularity:.5}', tuple(contours_list[0][0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            x_coord, y_coord = tuple(contours_list[0][0][0])
            cv2.putText(image, f'{fractal_dimension:.5}', (x_coord, y_coord-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            break_out = False
            for point in points:
                if np.sqrt((point[0]-(1.0*width/2))**2+(point[1]-(1.0*height/2))**2) > 1100:
                    break_out = True
                    break
            if break_out == True:
                break
                
            # Display the original and thresholded images
            image_filename = 'fracdim_'+image_path.split('/')[-1]
            if side in ['left','right']:
                image_filename = side + '_' + image_filename
            if not args['--include_holes']:
                image_filename = 'nh_' + image_filename
            cv2.imwrite('PROCESSED' + '/' + image_filename, image)#final_image)
            output += f'{frame:>9} {total_area:16e} {total_perimeter:16e} {circularity:16e} {fractal_dimension:16e}\r\n'
            
            plt.figure(image_index)
            plt.scatter(logbsr, logN)
            plt.plot(logbsr, fractal_dimension*np.array(logbsr)+intercept, label=r'$%3.2f log(size) + %3.2f$' % (fractal_dimension, intercept) )
            plt.title(image_filename)
            plt.ylabel('log(N)')
            plt.xlabel('log(size)')
            #plt.text(min(logbsr), min(logN), f'$d={fractal_dimension:3.2}$')
            plt.legend(loc=1)
            plt.savefig('PROCESSED' + '/' + image_filename.replace('.jpg','_plt.jpg') )

            # generate a physarum_only image
            physarum_only_image = cv2.bitwise_and(physarum_only_image, physarum_only_image, mask=physarum_mask)
            cv2.imwrite('PROCESSED' + '/' + image_filename.replace('fracdim', 'physarum'), physarum_only_image)
            
        fileout.write(output)
        fileout_frac.write(output_frac)
