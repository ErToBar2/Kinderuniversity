import cv2
import numpy as np
import pygame
from ultralytics import YOLO

class PoseEstimationRitual:
    def __init__(self):
        # Load the YOLO model with the specified weights for pose estimation
        self.model = YOLO("yolo-Weights\yolo11x-pose.pt")
        self.frame_counter = 0
        self.keypoint_names = [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
        ]
        self.feature_results = None
        # Statue's arms (fixed points)
        self.statue_left = {'shoulder': (940, 458), 'elbow': (908, 481), 'wrist': (866, 447)}
        self.statue_right = {'shoulder': (1003, 461), 'elbow': (1042, 488), 'wrist': (1087, 453)}
        self.statue_polyline = [
                (self.statue_left['shoulder'], self.statue_left['elbow']),
                (self.statue_left['elbow'], self.statue_left['wrist']),
                (self.statue_right['shoulder'], self.statue_right['elbow']),
                (self.statue_right['elbow'], self.statue_right['wrist']),
                (self.statue_left['shoulder'], self.statue_right['shoulder'])  # Connect the shoulders
            ]

        self.circle_closed = False
        self.ritualsteps = 1

        self.ritual_energy_charge_step2 = 0
        self.ritual_overlay_alpha_step2 = 0
        self.arms_up = []

#### ADJUST VALUES BELOW ####
        self.min_distance = 90
        self.vertical_threshold = 100




    def update_ritualsteps(self, ritualsteps):
        """Update the current ritual step."""
        self.ritualsteps = ritualsteps


    def process_frame(self, frame, pygame_surface, target_point=None):
        
        results = self.model(frame, verbose=False, imgsz=(1920, 1088), conf=0.5)
        self.frame_counter += 1

        # Perform visualization every 2 frames
        
        people_keypoints = self.extract_keypoints(results)
        # pre-process the important keypoints and overlay on screen
        self.overlay_game_frame(pygame_surface, people_keypoints)

        #print("ritualsteps:",  self.ritualsteps)
    # Ritual step 1:
        if self.ritualsteps == 1: # and self.frame_counter % 5 == 0:
            self.perform_matching_logic(pygame_surface)
            # Overlay the statue's polyline (arms and shoulders)
            for start_point, end_point in self.statue_polyline:
                pygame.draw.line(pygame_surface, (0, 0, 255), start_point, end_point, 5)
        
    # Ritual step 2: Check if all arms are held up
        if self.ritualsteps == 2:
            all_arms_up = self.check_arms_held_up()  # Check if all arms are up
            if len(self.arms_up_visualization) > 0:
                #print("xxxx Visualizing arms up", self.arms_up_visualization) 
                self.visualize_arms_up(self.arms_up_visualization, pygame_surface)


        annotated_webcam_frame = results[0].plot()

        # Return both the annotated webcam frame (for OpenCV) and the Pygame surface (for the game frame)
        return annotated_webcam_frame, pygame_surface




    def extract_keypoints(self, results):
        people_keypoints = []

        for i, result in enumerate(results):
            if result.keypoints is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()  # xy coordinates

                for person_keypoints in keypoints_xy:
                    if len(person_keypoints) >= 11:  # Ensure there are enough keypoints
                        wrists = [person_keypoints[9], person_keypoints[10]]  # Left and right wrists

                        # Store keypoints as a dictionary of coordinates
                        people_keypoints.append({
                            'shoulders': [person_keypoints[5], person_keypoints[6]],  # Shoulders (left, right)
                            'elbows': [person_keypoints[7], person_keypoints[8]],  # Elbows (left, right)
                            'wrists': wrists  # Wrists (left, right)
                        })
                        #print("people extract_keypoints: ", people_keypoints)
        
        return people_keypoints


    def overlay_game_frame(self, pygame_surface, people_keypoints):
        """Overlay the game frame with detected keypoints and connections for the game visualization,
        and store the visualized keypoints including person index as self.keypoints."""

        # Initialize or reset the keypoints list
        self.visualized_keypoints = []

        # Helper function to check if a keypoint is valid
        def is_valid_keypoint(kp):
            return len(kp) == 2 and not (abs(kp[0]) < 1e-2 and abs(kp[1]) < 1e-2)

        for idx, p in enumerate(people_keypoints):  # Correctly enumerate each person
            # Default to None if keypoints are missing
            shoulders = p.get('shoulders', [None, None])  # Left, right shoulders
            elbows = p.get('elbows', [None, None])        # Left, right elbows
            wrists = p.get('wrists', [None, None])        # Left, right wrists

            # Dictionary to store visualized keypoints for this person
            person_keypoints = {
                'person_idx': idx,  # Assign unique person index based on the loop
                'shoulders': [None, None],  # Placeholder for left, right shoulder
                'elbows': [None, None],     # Placeholder for left, right elbow
                'wrists': [None, None],     # Placeholder for left, right wrist
            }
            #
            #print(f"XXXXXXXXXXXXXXX Processing person {idx}, Shoulders: {shoulders}, Elbows: {elbows}, Wrists: {wrists}")  # Debugging

            # Draw left arm: Shoulder -> Elbow -> Wrist (if valid)
            if is_valid_keypoint(shoulders[0]):  # Left shoulder
                person_keypoints['shoulders'][0] = shoulders[0]
                if is_valid_keypoint(elbows[0]):  # Left elbow
                    person_keypoints['elbows'][0] = elbows[0]
                    # Draw from shoulder to elbow if both are valid
                    pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, shoulders[0])), tuple(map(int, elbows[0])), 5)
                    if is_valid_keypoint(wrists[0]):
                        person_keypoints['wrists'][0] = wrists[0]
                        pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, elbows[0])), tuple(map(int, wrists[0])), 5)
                    else:
                        pygame.draw.circle(pygame_surface, (255, 0, 0), tuple(map(int, elbows[0])), 10)  # Mark elbow
                elif is_valid_keypoint(wrists[0]):  # Left elbow is missing, but wrist is present
                    person_keypoints['wrists'][0] = wrists[0]
                    # Only connect shoulder directly to wrist if elbow is missing
                    pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, shoulders[0])), tuple(map(int, wrists[0])), 5)
                else:
                    pygame.draw.circle(pygame_surface, (255, 0, 0), tuple(map(int, shoulders[0])), 10)  # Mark shoulder

            # Draw right arm: Shoulder -> Elbow -> Wrist (if valid)
            if is_valid_keypoint(shoulders[1]):  # Right shoulder
                person_keypoints['shoulders'][1] = shoulders[1]
                if is_valid_keypoint(elbows[1]):  # Right elbow
                    person_keypoints['elbows'][1] = elbows[1]
                    # Draw from shoulder to elbow if both are valid
                    pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, shoulders[1])), tuple(map(int, elbows[1])), 5)
                    if is_valid_keypoint(wrists[1]):
                        person_keypoints['wrists'][1] = wrists[1]
                        pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, elbows[1])), tuple(map(int, wrists[1])), 5)
                    else:
                        pygame.draw.circle(pygame_surface, (255, 0, 0), tuple(map(int, elbows[1])), 10)  # Mark elbow
                elif is_valid_keypoint(wrists[1]):  # Right elbow is missing, but wrist is present
                    person_keypoints['wrists'][1] = wrists[1]
                    # Only connect shoulder directly to wrist if elbow is missing
                    pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, shoulders[1])), tuple(map(int, wrists[1])), 5)
                else:
                    pygame.draw.circle(pygame_surface, (255, 0, 0), tuple(map(int, shoulders[1])), 10)  # Mark shoulder

            # Connect shoulders if both are valid
            if is_valid_keypoint(shoulders[0]) and is_valid_keypoint(shoulders[1]):
                pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, shoulders[0])), tuple(map(int, shoulders[1])), 5)

            # Append the person keypoints to the final list
            self.visualized_keypoints.append(person_keypoints)
            ### print("visualized keypoints: ", self.visualized_keypoints)

        return pygame_surface

    def interpolate_wrist(self, wrist, elbow):
        # Interpolate wrist position to be 10% further away from the elbow
        wrist_np = np.array(wrist)
        elbow_np = np.array(elbow)
        direction = wrist_np - elbow_np
        new_wrist = wrist_np + 0.1 * direction
        return tuple(map(int, new_wrist))



    def perform_matching_logic(self, pygame_surface):
        """
        Perform the wrist matching logic and visualize either matched wrists (lines)
        or unmatched wrists (circles). Additionally, draw a thin blue line between matched pairs.
        """

        # Helper function to check if a wrist is valid
        def is_valid_keypoint(kp):
            if kp is None:
                return False
            return len(kp) == 2 and not (abs(kp[0]) < 1e-2 and abs(kp[1]) < 1e-2)

        # Create a new list to store all wrist data in a simplified structure
        all_wrists = []  # List of tuples (index, (x, y)) for people and the statue

        # Add statue wrists if valid (using index 99 for both left and right wrists)
        if is_valid_keypoint(self.statue_left['wrist']):
            x, y = map(int, self.statue_left['wrist'])  # Ensure coordinates are integers
            all_wrists.append((99, (x, y)))
        if is_valid_keypoint(self.statue_right['wrist']):
            x, y = map(int, self.statue_right['wrist'])  # Ensure coordinates are integers
            all_wrists.append((99, (x, y)))

        # Add all the people's wrists, assuming they've been processed by overlay_game_frame
        for idx, p in enumerate(self.visualized_keypoints):
            left_wrist = p['wrists'][0]
            right_wrist = p['wrists'][1]
            left_elbow = p['elbows'][0]
            right_elbow = p['elbows'][1]
            if is_valid_keypoint(left_wrist) and is_valid_keypoint(left_elbow):
                interpolated_left_wrist = self.interpolate_wrist(left_wrist, left_elbow)
                all_wrists.append((idx, interpolated_left_wrist))
            if is_valid_keypoint(right_wrist) and is_valid_keypoint(right_elbow):
                interpolated_right_wrist = self.interpolate_wrist(right_wrist, right_elbow)
                all_wrists.append((idx, interpolated_right_wrist))

        # Initialize open_wrists with the newly structured wrist list
        open_wrists = all_wrists[:]

        # Perform the wrist matching (minimum distance check)
        min_distance = self.min_distance
        num_wrists = len(all_wrists)

        # Loop through all unique pairs of wrists
        for i in range(num_wrists):
            for j in range(i + 1, num_wrists):  # Ensure we only compare each unique pair once
                idx1, wrist1 = all_wrists[i]
                idx2, wrist2 = all_wrists[j]

                # Prevent wrists from the same person (i.e., idx1 should not be equal to idx2)
                if idx1 != idx2:
                    # Calculate the distance between the two wrists
                    distance = np.linalg.norm(np.array(wrist1) - np.array(wrist2))
                    #print(f"Checking wrists {idx1} and {idx2}, distance: {distance}")

                    if distance < min_distance:
                        # Wrists are matched, draw lines and remove from open_wrists
                        pygame.draw.line(pygame_surface, (0, 255, 0), wrist1, wrist2, 5)
                        pygame.draw.line(pygame_surface, (0, 0, 255), wrist1, wrist2, 2)

                        # Remove matched wrists from open_wrists
                        open_wrists = [
                            (idx, w) for (idx, w) in open_wrists
                            if not (w == wrist1 and idx == idx1) and not (w == wrist2 and idx == idx2)
                        ]

        # Visualize unmatched wrists
        for idx, wrist in open_wrists:
            pygame.draw.circle(pygame_surface, (255, 255, 0), wrist, min_distance/2, 5)

        # Optionally: store whether all wrists are matched (circle closed)
        self.circle_closed = len(open_wrists) == 0


    def draw_matched_wrist(self, pygame_surface, wrist1, wrist2):
        """Draw a thin blue line between matched wrists."""
        pygame.draw.line(pygame_surface, (0, 0, 255), tuple(map(int, wrist1)), tuple(map(int, wrist2)), 2)  # Blue line


    def is_circle_closed(self):
        return self.circle_closed




    def check_arms_held_up(self):
        """
        Check if all visible arms are held up, return True or False.
        Visualize the arms that are held up even if not all are up.
        """
        arms_up = []
        self.arms_up_visualization = []
        arms_not_up = []

        # Define a threshold for how "vertical" the arm should be
        vertical_threshold = self.vertical_threshold  # Adjust this value to be more or less strict

        # Iterate through each person's keypoints
        for person_keypoints in self.visualized_keypoints:
            shoulders = person_keypoints.get('shoulders', [None, None])
            elbows = person_keypoints.get('elbows', [None, None])
            wrists = person_keypoints.get('wrists', [None, None])


            for i in range(2):  # 0: Left arm, 1: Right arm
                if wrists[i] is None or elbows[i] is None or shoulders[i] is None:
                    continue  # Skip if any keypoint is missing
                
                # Print keypoints for debugging
                print(f"Arm {i}: Shoulder: {shoulders[i]}, Elbow: {elbows[i]}, Wrist: {wrists[i]}")

                # Check if the arm is up (wrist < elbow < shoulder on y-axis)
                arm_is_up = wrists[i][1] < elbows[i][1] < shoulders[i][1]

                # Check if the arm is roughly vertical (small x-difference between shoulder, elbow, wrist)
                vertical_check = abs(shoulders[i][0] - elbows[i][0]) < vertical_threshold and \
                                abs(elbows[i][0] - wrists[i][0]) < vertical_threshold
                
                if arm_is_up and vertical_check:
                    arms_up.append(i)
                    self.arms_up_visualization.append((shoulders[i], elbows[i], wrists[i]))
                else:
                    arms_not_up.append(i)
        
        # print("arms_up: ", arms_up)
        # print("arms_not_up: ", arms_not_up)

        # all_arms_up is True if there are no arms in the "not up" list and some arms are detected as "up"
        all_arms_up = len(arms_not_up) == 0 and len(arms_up) > 0
        # print("All arms up: ", all_arms_up)

        return all_arms_up


    def visualize_arms_up(self, arms_up_list, pygame_surface):
        """
        Visualize the arms that are held up on the Pygame surface.
        arms_up_list: List of indices for the arms that are considered "up".
        pygame_surface: The surface where we draw the arms.
        """
        color = (164, 255, 255)  # Light blue color for arms that are up
        thickness = 7  # Line thickness
        arms_up_visualization = arms_up_list
        for keypoints in arms_up_visualization:
            shoulder, elbow, wrist = keypoints

            # Draw shoulder to elbow line
            shoulder = tuple(map(int, shoulder))
            elbow = tuple(map(int, elbow))
            wrist = tuple(map(int, wrist))

            # Now draw the lines
            pygame.draw.line(pygame_surface, color, shoulder, elbow, thickness)
            pygame.draw.line(pygame_surface, color, elbow, wrist, thickness)

