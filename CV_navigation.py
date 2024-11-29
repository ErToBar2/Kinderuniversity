import cv2
import numpy as np
from ultralytics import YOLO

class CVNavigation:
    def __init__(self, control_inputs, width_size_video, height_size_video):
        self.model = YOLO('yolo-Weights\yolov8x-worldv2.pt')
        self.control_inputs = control_inputs
        self.width_size_video = width_size_video
        self.height_size_video = height_size_video
        self.frame_count = 0
        self.colors = {
            'black hat': (0, 0, 0),          # Black color for Player 1
            'white hat': (255, 255, 255),    # White color for Player 2

            'yellow scissors': (255, 255, 255),
            'blue lighter': (255, 255, 0),
            'wooden spoon with shovel': (0, 255, 0),
            'branch': (0, 255, 255),

            'book': (255, 0, 0),
            'necklace': (0, 0, 255),
            'frisbee': (0, 255, 0)
            # Add other classes as needed
        }

        self.detection_classes = []
        self.label_mapping = {
            'hat': 'Hat',
            'yellow scissors': 'Sharp metal part',
            'wooden spoon with shovel': 'Torch',
            'blue lighter': 'Lighter',
            'branch': 'Branch',
            'book': 'Rune',
            'necklace': 'Medallion',
            'frisbee': 'Rune'
            # Add other classes as needed
        }

        self.confidence_thresholds = {
            'hat': 0.1,
            'yellow scissors': 0.5,
            'blue lighter': 0.7,
            'wooden spoon with shovel': 0.4,
            'branch': 0.1,
            'book': 0.10,
            'necklace': 0.3,
            'frisbee': 0.1
            
            # Add other classes as needed
        }
        self.cut_loose = False
        self.use_lighter = False
        self.use_stick = False
        self.medallion_back = False
        self.checked_frame = 9  # Check key signals every 15 frames
        self.players_positions = {'player1': [0, 0], 'player2': [0, 0]}  # Initialize with default positions
        self.detected_objects = []


    def get_control_inputs(self):
        return self.control_inputs

    def get_cut_loose(self):
        return self.cut_loose 

    def get_use_lighter(self):
        return self.use_lighter

    def get_use_stick(self):
        return self.use_stick

    def get_medallion_back(self):
        return self.medallion_back

    def set_checked_frame(self, value):
        self.checked_frame = value

    def update_player_positions(self, player1_position, player2_position):
        self.players_positions['player1'] = player1_position
        self.players_positions['player2'] = player2_position
        

    def classify_hat_color(self, roi):
        """
        Classify the hat color as 'black hat' or 'white hat' based on average brightness.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        # Define a threshold to differentiate between black and white
        return 'black hat' if avg_brightness < 100 else 'white hat'  # Adjust threshold as needed

    def set_detection_classes(self, classes):
        self.detection_classes = classes
        self.model.set_classes(classes)


    def classify_rune_color(self, roi):
        """
        Fast color classifier that analyzes the center area of the ROI and determines the dominant color (R, G, or B).
        """
        # Get the dimensions of the ROI
        height, width, _ = roi.shape

        # Define the center region as a fraction of the ROI's size (e.g., 50% of the center)
        center_fraction = 0.5
        x1 = int(width * (1 - center_fraction) / 2)
        y1 = int(height * (1 - center_fraction) / 2)
        x2 = int(width * (1 + center_fraction) / 2)
        y2 = int(height * (1 + center_fraction) / 2)
        
        # Extract the center area of the ROI
        center_roi = roi[y1:y2, x1:x2]

        # Calculate the average B, G, R values in the center area
        average_color = np.mean(center_roi, axis=(0, 1))  # This returns the average B, G, R values
        
        # Determine the dominant color by comparing R, G, and B
        b, g, r = average_color  # OpenCV uses BGR format
        g *= 1.3  #
        print("average_color", r, g, b)
        if max(r, g, b) == r:
            print("red, with:", r )
            return 'red'
        elif max(r, g, b) == g:
            print("green, with:", g )
            return 'green'
        elif max(r, g, b) == b:
            print("blue, with:", b )
            return 'blue'

        # Return None if no dominant color is found (you can add a threshold if needed)
        return None



    def get_detected_objects(self):
        """
        Get the list of currently detected objects with their relative positions and labels.
        This can be accessed in the main loop.
        """
        return self.detected_objects

### main process frame function ###


    def process_frame(self, frame):
        height, width, _ = frame.shape
        imgsz = [self.width_size_video, self.height_size_video]
        
        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            stream=True,
            verbose=False,
            conf=0.1
        )
        
        self.use_lighter = False
        ### hoptfix self.use_stick = False
        # Initialize separate detection counts for black and white hats
        
        detection_counts_black = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        detection_counts_white = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.detected_objects = []

        # Define regions for navigation
        left_column_width = int(0.2 * width)
        right_column_start = int(0.8 * width)
        top_line_height = int(0.2 * height)
        bottom_line_start = int(0.8 * height)

        # Define colors and labels
        regions = [
            {'name': 'Left', 'rect': ((0, 0), (left_column_width, height)), 'color': (255, 0, 0)},
            {'name': 'Right', 'rect': ((right_column_start, 0), (width, height)), 'color': (0, 255, 0)},
            {'name': 'Up', 'rect': ((0, 0), (width, top_line_height)), 'color': (0, 0, 255)},
            {'name': 'Down', 'rect': ((0, bottom_line_start), (width, height)), 'color': (255, 255, 0)}
        ]

        # Draw rectangles and labels for regions
        # Draw rectangles and adjust the position of labels for regions
        for region in regions:
            (x1, y1), (x2, y2) = region['rect']
            color = region['color']
            name = region['name']
            
            # Draw the rectangle for the region (unchanged)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Adjust label positions based on the region
            if name == 'Up':
                # Centered horizontally, near the top of the region
                label_x = (x2 - x1) // 2  # Center horizontally
                label_y = y1 + 30         # Position near the top
            elif name == 'Down':
                # Centered horizontally, near the bottom of the region
                label_x = (x2 - x1) // 2  # Center horizontally
                label_y = y2 - 10         # Position near the bottom
            elif name == 'Left':
                # Centered vertically, near the left side
                label_x = x1 + 10         # Position near the left edge
                label_y = (y2 + y1) // 2  # Center vertically
            elif name == 'Right':
                # Centered vertically, near the right side
                label_x = x2 - 70         # Position near the right edge (adjust for text width)
                label_y = (y2 + y1) // 2  # Center vertically
            
            # Add the label text
            cv2.putText(
                frame,
                name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Iterate through detection results
        for result in results:
            for bbox in result.boxes:
                confidence = bbox.conf.item()
                class_idx = int(bbox.cls)

                if class_idx >= len(self.model.names):
                    continue  # Skip invalid class indices

                class_name = self.model.names[class_idx]

                # Get the specific confidence threshold for the detected class
                threshold = self.confidence_thresholds.get(class_name, 0.2)  # Default threshold

                if confidence < threshold:
                    continue  # Skip detections below the threshold

                if class_name == 'hat':
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    # Ensure coordinates are within frame boundaries
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue  # Skip if ROI is empty
                    hat_color = self.classify_hat_color(roi)

                    # Update display label and color based on classification
                    label = f"Player 1 {confidence:.2f}" if hat_color == 'black hat' else f"Player 2 {confidence:.2f}"

                    color = self.colors.get(hat_color, (0, 255, 255))  # Default to cyan
                else:
                    # Get display label and color for non-hat classes
                    display_label = self.label_mapping.get(class_name, class_name)
                    label = f"{display_label} {confidence:.2f}"
                    color = self.colors.get(class_name, (0, 255, 255))  # Default to cyan

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                ### Send signals to the game ###
                if class_name == 'hat':
                    # Determine the center point of the hat
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Check if the center point is within each region
                    if center_x < left_column_width:
                        if hat_color == 'black hat':
                            detection_counts_black['left'] += 1
                        elif hat_color == 'white hat':
                            detection_counts_white['left'] += 1

                    if center_x > right_column_start:
                        if hat_color == 'black hat':
                            detection_counts_black['right'] += 1
                        elif hat_color == 'white hat':
                            detection_counts_white['right'] += 1

                    if center_y < top_line_height:
                        if hat_color == 'black hat':
                            detection_counts_black['up'] += 1
                        elif hat_color == 'white hat':
                            detection_counts_white['up'] += 1

                    if center_y > bottom_line_start:
                        if hat_color == 'black hat':
                            detection_counts_black['down'] += 1
                        elif hat_color == 'white hat':
                            detection_counts_white['down'] += 1


                if class_name == 'frisbee':
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    # Ensure coordinates are within frame boundaries
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)

                    # Crop the region of interest (ROI) for color classification
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue  # Skip if ROI is empty

                    # Classify the rune color
                    rune_color = self.classify_rune_color(roi)
                    #print(f"Detected 'book' classified as rune color: {rune_color} at ({x1}, {y1})")

                    # Calculate the relative position
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height

                    # Store detected object information
                    self.detected_objects.append({
                        'label': f'rune_{rune_color}',
                        'relative_position': (center_x, center_y),
                        'confidence': confidence
                    })
                    # print("Detected rune color:", rune_color)
                    # print("Detected rune position:", (center_x, center_y))  # Relative position

                else:
                    # Handle other classes like 'yellow scissors', 'blue lighter', etc.
                    if class_name == 'yellow scissors':
                        if self.players_positions['player2'] == [9, 5]:
                            self.control_inputs['t'] = True
                            print("control_inputs['t'] set to:", self.control_inputs['t'])
                            self.cut_loose = True

                    elif class_name == 'blue lighter':
                        if (
                            self.players_positions['player1'] == [4, 7] or
                            self.players_positions['player2'] == [4, 7]
                        ):
                            self.control_inputs['t'] = True
                            self.use_lighter = True

                    elif class_name in ['wooden spoon with shovel']:
                        if (
                            self.players_positions['player1'] == [4, 7] or
                            self.players_positions['player2'] == [4, 7]
                        ):
                            self.control_inputs['z'] = True
                            self.use_stick = True
                    
                    elif class_name in ['necklace']:
                        if (
                            self.players_positions['player1'] == [9, 7] or
                            self.players_positions['player2'] == [9, 7]
                        ):
                            self.control_inputs['t'] = True
                            self.medallion_back = True
                    
                    
     
                    
        

        # Aggregate key signals every 15 frames
        if self.frame_count % self.checked_frame == 0:
            # Reset all controls first for Player 1
            for key in ['left', 'right', 'up', 'down']:
                self.control_inputs[key] = False

            # Set control inputs based on detection counts for Player 1
            for direction in ['left', 'right', 'up', 'down']:
                if detection_counts_black[direction] > 0:
                    self.control_inputs[direction] = True  # Simulate key press

            # Reset all controls first for Player 2
            for key in ['a', 'd', 'w', 's']:
                self.control_inputs[key] = False

            # Map directions to keys for Player 2
            key_mapping = {'left': 'a', 'right': 'd', 'up': 'w', 'down': 's'}

            # Set control inputs based on detection counts for Player 2
            for direction in ['left', 'right', 'up', 'down']:
                if detection_counts_white[direction] > 0:
                    self.control_inputs[key_mapping[direction]] = True  # Simulate key press
        
        
        self.frame_count += 1
        
        return frame
