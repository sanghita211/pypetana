"""
Video to PNG Frame Extractor. Can be used with multiple videos.

Usage:
  video_to_frames.py [options] <output_folder> <video_paths>... 

Arguments:
  <video_path>      Path to the input video file.
  <output_folder>   Path to the output folder for PNG images.
  --prefix=<str>    Override the PNG Frame prefix.
  --extension=<ext> Override default extension [default: png]
"""

import os
import cv2
from docopt import docopt

def extract_frames(video_path, output_folder, prefix='frame_', extension='png'):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was not read successfully, exit the loop
        if not ret:
            break

        # Construct the output filename
        print(f'{extension}')
        output_filename = os.path.join(output_folder, f"{prefix}{frame_count:03d}.{extension}")

        # Save the frame as a PNG image
        cv2.imwrite(output_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")

def main():
    arguments = docopt(__doc__)
    video_paths = arguments['<video_paths>']
    output_folder = arguments['<output_folder>']
    prefix = arguments['--prefix']
    extension = arguments['--extension']

    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: The video file '{video_path}' does not exist.")
            return
        
        prefix = video_path.split('.')[0]
        if arguments['--prefix']:
            prefix = arguments['--prefix']
        extract_frames(video_path, output_folder, prefix=prefix, extension=extension)

if __name__ == "__main__":
    main()


