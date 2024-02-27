import cv2
import numpy as np
import sys
import os

# Check if the input file path was provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python video_delta.py <input_file_path>")
    exit()

# Input video path from the first command-line argument
input_video_path = sys.argv[1]

# Generate the output video path by appending '_heatmap' before the file extension
base_name, extension = os.path.splitext(input_video_path)
output_video_path = f"{base_name}_heatmap{extension}"

cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {input_video_path}.")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
accumulated_diffs = np.zeros_like(prev_frame_gray, dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
    accumulated_diffs += frame_diff.astype(np.float32)
    prev_frame_gray = frame_gray

cv2.normalize(accumulated_diffs, accumulated_diffs, 0, 255, cv2.NORM_MINMAX)
heatmap = np.uint8(accumulated_diffs)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (prev_frame_gray.shape[1], prev_frame_gray.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_color = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(frame_gray_color, 0.7, heatmap_color, 0.3, 0)
    out.write(overlay)

cap.release()
out.release()
print(f"Processing completed. The output video is saved as {output_video_path}.")
