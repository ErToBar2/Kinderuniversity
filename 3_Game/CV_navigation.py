# CV_navigation.py
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO


class CVNavigation:
    def __init__(self, control_inputs, width_size_video, height_size_video, debug=True):
        # ---------------- Model / device setup ----------------
        self.model = YOLO(r'yolo-Weights\yolov8x-worldv2.pt')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model.to(self.device)
        except Exception as e:
            print("[CVN] Warning: model.to(device) failed:", e)

        # Set the *full* universe of classes ONCE to avoid CLIP device churn mid-game.
        self.class_universe = [
            'hat',
            'yellow scissors',
            'blue lighter',
            'wooden spoon with shovel',
            'necklace',
            'frisbee',
            'branch',
            'book',
        ]
        try:
            self.model.set_classes(self.class_universe)
            print("[CVN] set_classes() done once at init on device:", self.device)
        except Exception as e:
            print("[CVN] set_classes at init failed on", self.device, "-> trying CPU fallback:", e)
            try:
                self.model.to('cpu')
                self.model.set_classes(self.class_universe)
                self.model.to(self.device)
                print("[CVN] set_classes succeeded via CPU fallback; model returned to", self.device)
            except Exception as e2:
                print("[CVN] FATAL: set_classes CPU fallback also failed:", e2)

        # ---------------- External inputs / sizes ----------------
        self.control_inputs = control_inputs
        self.width_size_video = width_size_video
        self.height_size_video = height_size_video

        # ---------------- State ----------------
        self.frame_count = 0
        self.checked_frame = 9  # how often we aggregate joystick signals
        self.players_positions = {'player1': [0, 0], 'player2': [0, 0]}
        self.detected_objects = []

        # Action flags exposed to game
        self.cut_loose = False
        self.use_lighter = False
        self.use_stick = False
        self.medallion_back = False

        # Timers for "hold to activate" (in seconds)
        self.hold_requirements = {
            'yellow scissors': 2.0,
            'blue lighter': 2.0,
            'wooden spoon with shovel': 2.0,
            'necklace': 2.0,
        }
        self.hold_accum = {k: 0.0 for k in self.hold_requirements.keys()}
        # Decay only when AT the right location (helps with flicker). Leaving the spot = hard reset.
        self.decay_rate_at_location = 1.0  # seconds of decay per real second when object flickers at the correct spot

        # Draw / debug
        self.debug = debug
        self.font_scale_label = 0.7
        self.font_scale_timer = 0.8
        self.font_thickness = 2

        # Whitelist of classes for current level (we no longer call model.set_classes here)
        self.detection_classes = set()

        # Colors and labels
        self.colors = {
            'black hat': (0, 0, 0),
            'white hat': (255, 255, 255),
            'yellow scissors': (255, 255, 255),
            'blue lighter': (255, 255, 0),
            'wooden spoon with shovel': (0, 255, 0),
            'branch': (0, 255, 255),
            'book': (255, 0, 0),
            'necklace': (0, 0, 255),
            'frisbee': (0, 255, 0),
        }
        self.label_mapping = {
            'hat': 'Hat',
            'yellow scissors': 'Sharp metal part',
            'wooden spoon with shovel': 'Torch',
            'blue lighter': 'Lighter',
            'branch': 'Branch',
            'book': 'Rune',
            'necklace': 'Medallion',
            'frisbee': 'Rune',
        }
        self.confidence_thresholds = {
            'hat': 0.10,
            'yellow scissors': 0.50,
            'blue lighter': 0.70,
            'wooden spoon with shovel': 0.40,
            'branch': 0.10,
            'book': 0.10,
            'necklace': 0.30,
            'frisbee': 0.10,
        }

        # timebase
        self._prev_time = time.time()

    # --------------- Public getters ---------------
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

    def get_detected_objects(self):
        return self.detected_objects

    # --------------- External config ---------------
    def set_checked_frame(self, value):
        self.checked_frame = int(value)

    def update_player_positions(self, player1_position, player2_position):
        self.players_positions['player1'] = list(player1_position)
        self.players_positions['player2'] = list(player2_position)

    def set_detection_classes(self, classes):
        """
        Update a whitelist of classes we care about right now.
        We do NOT call YOLO.set_classes here (we did it once at init).
        """
        self.detection_classes = set(classes or [])
        if self.debug:
            print(f"[CVN] detection whitelist -> {sorted(self.detection_classes)}")

    # --------------- Helpers ---------------
    @staticmethod
    def _center_crop(roi, fraction=0.5):
        """Return inner ROI by fraction (e.g. 0.5 = inner 50%)."""
        h, w = roi.shape[:2]
        x1 = int(w * (1 - fraction) / 2)
        y1 = int(h * (1 - fraction) / 2)
        x2 = int(w * (1 + fraction) / 2)
        y2 = int(h * (1 + fraction) / 2)
        return roi[y1:y2, x1:x2]

    def _hat_brightness(self, roi):
        """Return mean brightness from inner 50% of ROI to avoid background."""
        if roi.size == 0:
            return None
        center = self._center_crop(roi, 0.5)
        gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _is_at_required_location(self, class_name):
        """Gate timers: only start/continue when the player(s) stand at the right tile."""
        p1 = self.players_positions['player1']
        p2 = self.players_positions['player2']
        if class_name == 'yellow scissors':
            # player2 must be at [9,5]
            return p2 == [9, 5]
        if class_name in ('blue lighter', 'wooden spoon with shovel'):
            # either player at [4,7]
            return p1 == [4, 7] or p2 == [4, 7]
        if class_name == 'necklace':
            # either player at [9,7]
            return p1 == [9, 7] or p2 == [9, 7]
        return False

    def _class_timer_update(self, class_name, dt, present_now):
        """
        Update hold timers:
          - If NOT at required location -> hard reset to 0.
          - If at required location & present -> accumulate.
          - If at required location & NOT present -> gentle decay (flicker resistance).
        Returns (accum, required, at_location).
        """
        if class_name not in self.hold_requirements:
            return None, None, False

        at_loc = self._is_at_required_location(class_name)

        if not at_loc:
            if self.hold_accum[class_name] != 0.0 and self.debug:
                print(f"[CVN] Reset '{class_name}' timer: left required tile")
            self.hold_accum[class_name] = 0.0
            return self.hold_accum[class_name], self.hold_requirements[class_name], False

        # At location
        if present_now:
            self.hold_accum[class_name] = min(
                self.hold_requirements[class_name],
                self.hold_accum[class_name] + dt
            )
        else:
            # decay for flicker resistance while at right location
            self.hold_accum[class_name] = max(
                0.0,
                self.hold_accum[class_name] - self.decay_rate_at_location * dt
            )

        return self.hold_accum[class_name], self.hold_requirements[class_name], True

    @staticmethod
    def _draw_label_with_bg(img, text, org, font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """Draw text with a filled background box behind it."""
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = org
        # Box around text
        box_tl = (max(0, x), max(0, y - th - baseline))
        box_br = (max(0, x + tw + 6), max(0, y + 4))
        cv2.rectangle(img, box_tl, box_br, bg_color, -1)
        cv2.putText(img, text, (x + 3, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_centered_text_below(img, text, box, font_scale=0.8, thickness=2, color=(0, 0, 255), margin=8):
        """Draw red timer centered *below* the bounding box."""
        x1, y1, x2, y2 = box
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx = (x1 + x2) // 2
        tx = int(cx - tw // 2)
        ty = int(y2 + th + baseline + margin)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def classify_rune_color(self, roi):
        """Return 'red'/'green'/'blue' based on dominant channel in inner 50%."""
        if roi.size == 0:
            return None
        center = self._center_crop(roi, 0.5)
        avg_bgr = np.mean(center, axis=(0, 1))
        b, g, r = avg_bgr
        g *= 1.3  # small bias per your original code
        mx = max(r, g, b)
        if mx == r:
            return 'red'
        elif mx == g:
            return 'green'
        else:
            return 'blue'

    # --------------- Main per-frame ---------------
    def process_frame(self, frame, player1_pos=None, player2_pos=None):
        t_now = time.time()
        dt = max(1e-3, t_now - self._prev_time)  # seconds
        self._prev_time = t_now
        if player1_pos is not None and player2_pos is not None:
            self.players_positions['player1'] = list(player1_pos)
            self.players_positions['player2'] = list(player2_pos)



        height, width = frame.shape[:2]
        imgsz = [self.width_size_video, self.height_size_video]

        # Reset per-frame flags
        self.use_lighter = False
        self.use_stick = False
        self.cut_loose = False
        self.medallion_back = False
        self.detected_objects = []

        # Navigation regions
        left_column_width = int(0.2 * width)
        right_column_start = int(0.8 * width)
        top_line_height = int(0.2 * height)
        bottom_line_start = int(0.8 * height)

        # For direction aggregation
        detection_counts_black = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        detection_counts_white = {'left': 0, 'right': 0, 'up': 0, 'down': 0}

        # Run detection
        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            stream=False,
            verbose=False,
            conf=0.10  # global low conf, we apply per-class thresholds
        )

        # Collect boxes first so we can pick top-2 hats
        hat_boxes = []
        other_boxes = []

        if self.debug:
            print(f"[CVN] predict -> {len(results)} result(s)")

        for result in results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for b in result.boxes:
                confidence = float(b.conf.item())
                class_idx = int(b.cls)
                if class_idx >= len(self.model.names):
                    continue
                class_name = self.model.names[class_idx]

                # Whitelist filter (if set)
                if self.detection_classes and class_name not in self.detection_classes:
                    continue

                threshold = self.confidence_thresholds.get(class_name, 0.2)
                if confidence < threshold:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # clamp within frame
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                if class_name == 'hat':
                    hat_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': confidence, 'class_idx': class_idx
                    })
                else:
                    other_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': confidence, 'class_idx': class_idx, 'class_name': class_name
                    })

        # ---- Handle hats: choose top-2 by confidence and decide white/black robustly ----
        hat_boxes.sort(key=lambda d: d['conf'], reverse=True)
        hat_boxes = hat_boxes[:2]  # only two most confident

        # Compute brightness for each hat using inner 50% region
        hat_infos = []
        for hb in hat_boxes:
            roi = frame[hb['y1']:hb['y2'], hb['x1']:hb['x2']]
            bright = self._hat_brightness(roi)
            hat_infos.append({**hb, 'brightness': bright})

        # Decide labels: if 2 hats, brighter => white, darker => black
        hat_assignments = []
        if len(hat_infos) == 2 and all(h['brightness'] is not None for h in hat_infos):
            hats_sorted = sorted(hat_infos, key=lambda d: d['brightness'])
            darker, brighter = hats_sorted[0], hats_sorted[1]
            hat_assignments.append({**darker, 'hat_color': 'black hat'})
            hat_assignments.append({**brighter, 'hat_color': 'white hat'})
            if self.debug:
                print(f"[CVN] Two hats: brightness -> black:{darker['brightness']:.1f}, white:{brighter['brightness']:.1f}")
        elif len(hat_infos) == 1:
            h = hat_infos[0]
            color = 'black hat'
            if h['brightness'] is not None:
                if h['brightness'] > 125:
                    color = 'white hat'
                elif h['brightness'] < 95:
                    color = 'black hat'
                else:
                    color = 'white hat' if h['brightness'] >= 110 else 'black hat'
            hat_assignments.append({**h, 'hat_color': color})
            if self.debug and h['brightness'] is not None:
                print(f"[CVN] One hat: brightness {h['brightness']:.1f} -> {color}")

        # Draw regions labels (visual nav grid)
        regions = [
            {'name': 'Left', 'rect': ((0, 0), (left_column_width, height)), 'color': (255, 0, 0)},
            {'name': 'Right', 'rect': ((right_column_start, 0), (width, height)), 'color': (0, 255, 0)},
            {'name': 'Up', 'rect': ((0, 0), (width, top_line_height)), 'color': (0, 0, 255)},
            {'name': 'Down', 'rect': ((0, bottom_line_start), (width, height)), 'color': (255, 255, 0)}
        ]
        for region in regions:
            (rx1, ry1), (rx2, ry2) = region['rect']
            color = region['color']
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            if region['name'] == 'Up':
                tx, ty = (rx2 - rx1) // 2, ry1 + 30
            elif region['name'] == 'Down':
                tx, ty = (rx2 - rx1) // 2, ry2 - 10
            elif region['name'] == 'Left':
                tx, ty = rx1 + 10, (ry2 + ry1) // 2
            else:
                tx, ty = rx2 - 70, (ry2 + ry1) // 2
            cv2.putText(frame, region['name'], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # ---- Draw hats + update direction counts ----
        for hat in hat_assignments:
            x1, y1, x2, y2 = hat['x1'], hat['y1'], hat['x2'], hat['y2']
            hat_color = hat['hat_color']
            conf = hat['conf']
            color = self.colors.get(hat_color, (0, 255, 255))
            label = f"Player 1 {conf:.2f}" if hat_color == 'black hat' else f"Player 2 {conf:.2f}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label with background (above top-left)
            self._draw_label_with_bg(
                frame,
                label,
                org=(x1, y1 - 5),
                font_scale=self.font_scale_label,
                thickness=self.font_thickness,
                text_color=color,
                bg_color=(0, 0, 0)
            )
            if self.debug:
                print(f"[CVN] Draw hat label '{label}' at ({x1},{y1})")

            # Region-based nav
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cx < left_column_width:
                if hat_color == 'black hat':
                    detection_counts_black['left'] += 1
                else:
                    detection_counts_white['left'] += 1
            if cx > right_column_start:
                if hat_color == 'black hat':
                    detection_counts_black['right'] += 1
                else:
                    detection_counts_white['right'] += 1
            if cy < top_line_height:
                if hat_color == 'black hat':
                    detection_counts_black['up'] += 1
                else:
                    detection_counts_white['up'] += 1
            if cy > bottom_line_start:
                if hat_color == 'black hat':
                    detection_counts_black['down'] += 1
                else:
                    detection_counts_white['down'] += 1

        # ---- Process non-hat objects (items, runes, etc.) ----
        # Track which activation classes were actually detected this frame (at-location)
        present_at_location_this_frame = {k: False for k in self.hold_requirements.keys()}

        # First, draw and (maybe) update timers for detections we saw
        for ob in other_boxes:
            class_name = ob['class_name']
            x1, y1, x2, y2, confidence = ob['x1'], ob['y1'], ob['x2'], ob['y2'], ob['conf']
            display_label = self.label_mapping.get(class_name, class_name)
            color = self.colors.get(class_name, (0, 255, 255))
            label = f"{display_label} {confidence:.2f}"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw label (above tl)
            self._draw_label_with_bg(
                frame,
                label,
                org=(x1, y1 - 5),
                font_scale=self.font_scale_label,
                thickness=self.font_thickness,
                text_color=color,
                bg_color=(0, 0, 0)
            )
            if self.debug:
                print(f"[CVN] Draw item label '{label}' at ({x1},{y1})")

            # If class has a hold timer, ONLY run timer if at required location
            if class_name in self.hold_requirements:
                at_loc = self._is_at_required_location(class_name)
                if at_loc:
                    # present & at required tile -> accumulate
                    accum, req, _ = self._class_timer_update(class_name, dt, present_now=True)
                    present_at_location_this_frame[class_name] = True
                    timer_text = f"{accum:.1f}/{int(req)}s"
                    # Draw timer BELOW the box (red)
                    self._draw_centered_text_below(
                        frame, timer_text, (x1, y1, x2, y2),
                        font_scale=self.font_scale_timer, thickness=self.font_thickness, color=(0, 0, 255), margin=8
                    )
                    if self.debug:
                        print(f"[CVN] Timer '{class_name}' (RUN) {timer_text} under box ({x1},{y1},{x2},{y2})")
                else:
                    # Not at location -> hard reset and (optionally) show '0/i s' if you want
                    if self.hold_accum[class_name] != 0.0:
                        if self.debug:
                            print(f"[CVN] Timer '{class_name}' -> hard reset (not at required tile)")
                    self.hold_accum[class_name] = 0.0
                    # (no timer text drawn if not at location)

            # Special handling
            if class_name == 'frisbee':
                # Rune color classification for your level 5 mechanic
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    rune_color = self.classify_rune_color(roi)
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    self.detected_objects.append({
                        'label': f'rune_{rune_color}',
                        'relative_position': (center_x, center_y),
                        'confidence': confidence
                    })

        # Now, for classes that are at the correct tile but were NOT seen this frame:
        for cname in self.hold_requirements.keys():
            at_loc = self._is_at_required_location(cname)
            if at_loc and not present_at_location_this_frame[cname]:
                # Object flickered (at location; not detected this frame) -> decay
                accum, req, _ = self._class_timer_update(cname, dt, present_now=False)
                if self.debug and accum > 0.0:
                    print(f"[CVN] Timer '{cname}' (DECAY at location) -> {accum:.2f}/{int(req)}s")
            elif not at_loc:
                # Leaving the location already handled (hard reset) in loop above if seen; also do here for not-seen frames
                if self.hold_accum[cname] != 0.0:
                    if self.debug:
                        print(f"[CVN] Timer '{cname}' -> hard reset (not at required tile, no detection)")
                    self.hold_accum[cname] = 0.0

        # ---- Trigger game actions when timers are full AND positions match ----
        # 'yellow scissors' -> cut_loose at player2 on [9,5]
        if self.hold_accum.get('yellow scissors', 0.0) >= self.hold_requirements['yellow scissors']:
            if self.players_positions['player2'] == [9, 5]:
                self.control_inputs['t'] = True
                self.cut_loose = True
                self.hold_accum['yellow scissors'] = 0.0  # reset after firing
                if self.debug:
                    print("[CVN] >>> CUT LOOSE TRIGGERED")

        # 'blue lighter' -> use_lighter at [4,7] (either player)
        if self.hold_accum.get('blue lighter', 0.0) >= self.hold_requirements['blue lighter']:
            if (self.players_positions['player1'] == [4, 7]) or (self.players_positions['player2'] == [4, 7]):
                self.control_inputs['t'] = True
                self.use_lighter = True
                self.hold_accum['blue lighter'] = 0.0  # reset after firing
                if self.debug:
                    print("[CVN] >>> USE LIGHTER TRIGGERED")

        # 'wooden spoon with shovel' -> use_stick at [4,7] (either player)
        if self.hold_accum.get('wooden spoon with shovel', 0.0) >= self.hold_requirements['wooden spoon with shovel']:
            if (self.players_positions['player1'] == [4, 7]) or (self.players_positions['player2'] == [4, 7]):
                self.control_inputs['z'] = True
                self.use_stick = True
                self.hold_accum['wooden spoon with shovel'] = 0.0  # reset after firing
                if self.debug:
                    print("[CVN] >>> USE STICK TRIGGERED")

        # 'necklace' -> medallion back at [9,7] (either player)
        if self.hold_accum.get('necklace', 0.0) >= self.hold_requirements['necklace']:
            if (self.players_positions['player1'] == [9, 7]) or (self.players_positions['player2'] == [9, 7]):
                self.control_inputs['t'] = True
                self.medallion_back = True
                self.hold_accum['necklace'] = 0.0  # reset after firing
                if self.debug:
                    print("[CVN] >>> MEDALLION BACK TRIGGERED")

        # ---- Aggregate key signals every checked_frame ----
        if self.frame_count % self.checked_frame == 0:
            # Player 1 (arrow keys)
            for key in ['left', 'right', 'up', 'down']:
                self.control_inputs[key] = False
            for direction in ['left', 'right', 'up', 'down']:
                if detection_counts_black[direction] > 0:
                    self.control_inputs[direction] = True

            # Player 2 (WASD)
            for key in ['a', 'd', 'w', 's']:
                self.control_inputs[key] = False
            key_mapping = {'left': 'a', 'right': 'd', 'up': 'w', 'down': 's'}
            for direction in ['left', 'right', 'up', 'down']:
                if detection_counts_white[direction] > 0:
                    self.control_inputs[key_mapping[direction]] = True

        self.frame_count += 1
        return frame
