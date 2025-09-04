import cv2
import logging
import json
import numpy as np
import os

from core import (AppController, create_argument_parser, validate_yolo_args,
                  YOLO_VERSIONS, YOLO_VALID_SIZES_BY_VERSION, YOLO_VALID_TASKS_BY_VERSION, create_yolo_analyzer_params)
from analyzers import YoloBaseAnalyzer

# --- Logging Configuration ---
# Configure logging to display the time, log level, and message.
# This will output to the console where the script is running.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StandaloneApp:
    def __init__(self, args):
        self.args = args
        self.controller = AppController()
        self.window_name = "AI Video Analysis - Standalone"
        self.running = True

        # --- Additions for YOLO options ---
        self.yolo_task = args.yolo_task
        self.yolo_version = args.yolo_version
        self.yolo_size = args.yolo_size
        self.yolo_valid_sizes = YOLO_VALID_SIZES_BY_VERSION
        self.yolo_valid_tasks = YOLO_VALID_TASKS_BY_VERSION
        # --- End Additions ---

        # --- UI State ---
        self.ui_layout = {'buttons': {}, 'model_options': {}, 'yolo_version_options': {}, 'yolo_size_options': {}}
        self.yolo_trackbar_created = False
        self.sidebar_width = 220

        # --- JSON display state ---
        self.json_scroll_offset_y = 0
        self.json_line_height = 20
        self.json_padding = 10

    def _setup_analyzer(self):
        """Initializes the analyzer based on startup args."""
        if not self._reload_analyzer():
            return False
        return self.controller.analyzer is not None

    def _setup_video_source(self):
        try:
            if self.args.video:
                self.controller.start_video_file(self.args.video)
            else:
                self.controller.start_webcam()
        except IOError as e:
            logging.error(e)
            raise

    def _on_confidence_change(self, value):
        confidence = value / 100.0
        self.controller.set_confidence_threshold(confidence)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, rect in self.ui_layout.get('buttons', {}).items():
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    self._handle_button_click(name)
                    return
            
            for name, rect in self.ui_layout.get('model_options', {}).items():
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    self._handle_model_selection(name)
                    return
            # Add new handlers for YOLO options
            for version, rect in self.ui_layout.get('yolo_version_options', {}).items():
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    self._handle_yolo_version_selection(version)
                    return

            for size, rect in self.ui_layout.get('yolo_size_options', {}).items():
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    self._handle_yolo_size_selection(size)
                    return

            for task, rect in self.ui_layout.get('yolo_task_options', {}).items():
                if rect[0] < x < rect[2] and rect[1] < y < rect[3]:
                    self._handle_yolo_task_selection(task)
                    return

    def _handle_button_click(self, name):
        if name == 'Start/Pause/Resume':
            if self.controller.processing_state == 'running':
                self.controller.pause_processing()
            else:
                self.controller.start_processing()
        elif name == 'Stop':
            self.controller.stop_processing()
            self.running = False

    def _handle_model_selection(self, name):
        if self.args.model == name:
            return
        self.args.model = name
        self._reload_analyzer()

    def _handle_yolo_task_selection(self, task):
        if self.yolo_task != task:
            self.yolo_task = task
            self._reload_analyzer()

    def _reload_analyzer(self):
        """Sets the analyzer based on the current app state (args model, yolo version/size/pose)."""
        model_name = self.args.model
        analyzer_params = {}

        if model_name == "YOLO":
            analyzer_params = create_yolo_analyzer_params(
                version=self.yolo_version, size=self.yolo_size, task=self.yolo_task,
                confidence=self.controller.get_confidence_threshold()
            )
        elif model_name == "Gemini":
            analyzer_params = {"api_key": self.args.api_key}
        
        logging.info(f"Setting analyzer to {model_name} with params {analyzer_params}")
        analyzer_ready = self.controller.set_analyzer(model_name, **analyzer_params)
        if not analyzer_ready:
            logging.error(f"Failed to initialize {model_name} analyzer.")
            return False
        return True

    def _handle_yolo_version_selection(self, version):
        if self.yolo_version != version:
            self.yolo_version = version
            if self.yolo_task not in self.yolo_valid_tasks[self.yolo_version]:
                self.yolo_task = self.yolo_valid_tasks[self.yolo_version][0]
            # Reset size to a valid default for the new version
            if self.yolo_size not in self.yolo_valid_sizes[self.yolo_version]:
                self.yolo_size = self.yolo_valid_sizes[self.yolo_version][0]
            self._reload_analyzer()

    def _handle_yolo_size_selection(self, size):
        if self.yolo_size != size:
            self.yolo_size = size
            self._reload_analyzer()

    def _update_yolo_trackbar(self):
        is_yolo = isinstance(self.controller.analyzer, YoloBaseAnalyzer)
        if is_yolo and not self.yolo_trackbar_created:
            initial_confidence = int(self.controller.get_confidence_threshold() * 100)
            cv2.createTrackbar('Confidence', self.window_name, initial_confidence, 100, self._on_confidence_change)
            self.yolo_trackbar_created = True
        # Note: OpenCV doesn't have a way to hide/destroy a trackbar.
        # It will remain visible but will be ignored if the model is not YOLO.

    def _draw_sidebar(self, canvas, x_start, height):
        self.ui_layout['buttons'] = {}
        self.ui_layout['model_options'] = {}
        self.ui_layout['yolo_version_options'] = {}
        self.ui_layout['yolo_size_options'] = {}
        self.ui_layout['yolo_task_options'] = {}

        # Sidebar background
        cv2.rectangle(canvas, (x_start, 0), (canvas.shape[1], height), (40, 40, 40), -1)
        
        y_pos = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- Draw Processing Controls ---
        state = self.controller.processing_state
        btn1_text = "Pause" if state == 'running' else "Resume" if state == 'paused' else "Start"
        
        btn_rects = {
            'Start/Pause/Resume': (x_start + 20, y_pos, x_start + self.sidebar_width - 20, y_pos + 40),
            'Stop': (x_start + 20, y_pos + 50, x_start + self.sidebar_width - 20, y_pos + 90) # q
        }
        
        # Draw Start/Pause/Resume button
        cv2.rectangle(canvas, (btn_rects['Start/Pause/Resume'][0], btn_rects['Start/Pause/Resume'][1]), (btn_rects['Start/Pause/Resume'][2], btn_rects['Start/Pause/Resume'][3]), (80, 80, 80), -1)
        cv2.putText(canvas, f"{btn1_text} (p)", (x_start + 45, y_pos + 28), font, 0.7, (255, 255, 255), 1)
        self.ui_layout['buttons']['Start/Pause/Resume'] = btn_rects['Start/Pause/Resume']

        # Draw Stop button
        cv2.rectangle(canvas, (btn_rects['Stop'][0], btn_rects['Stop'][1]), (btn_rects['Stop'][2], btn_rects['Stop'][3]), (80, 80, 80), -1)
        cv2.putText(canvas, "Stop (q)", (x_start + 60, y_pos + 78), font, 0.7, (255, 255, 255), 1)
        self.ui_layout['buttons']['Stop'] = btn_rects['Stop']

        y_pos += 140

        # --- Draw Model Selection ---
        cv2.putText(canvas, "Models:", (x_start + 10, y_pos), font, 0.7, (255, 255, 255), 1)
        y_pos += 30

        models = ["YOLO", "DeepFace", "Gemini"]
        current_model_name = self.args.model

        for model_name in models:
            is_selected = model_name == current_model_name
            color = (0, 255, 0) if is_selected else (220, 220, 220)
            btn_rect = (x_start + 20, y_pos - 20, x_start + self.sidebar_width - 20, y_pos + 15)
            
            radius = 8
            center_x = btn_rect[0] + 15
            center_y = y_pos
            cv2.circle(canvas, (center_x, center_y), radius, (220, 220, 220), 1)
            if is_selected:
                cv2.circle(canvas, (center_x, center_y), radius - 3, color, -1)

            cv2.putText(canvas, model_name, (center_x + 15, y_pos + 5), font, 0.7, color, 1)
            self.ui_layout['model_options'][model_name] = btn_rect
            y_pos += 45

        # --- Draw YOLO specific options if YOLO is selected ---
        if current_model_name == "YOLO":
            y_pos += 20
            cv2.line(canvas, (x_start + 10, y_pos), (x_start + self.sidebar_width - 10, y_pos), (80, 80, 80), 1)
            y_pos += 30
            cv2.putText(canvas, "YOLO Options:", (x_start + 10, y_pos), font, 0.7, (255, 255, 255), 1)
            y_pos += 30

            # Task selection
            cv2.putText(canvas, "Task:", (x_start + 10, y_pos), font, 0.7, (255, 255, 255), 1)
            y_pos += 30
            tasks = self.yolo_valid_tasks.get(self.yolo_version, [])
            item_height = 30
            for i, task in enumerate(tasks):
                is_selected = task == self.yolo_task
                color = (0, 255, 0) if is_selected else (220, 220, 220)
                col = i % 2
                row = i // 2
                base_x = x_start + 25 + (col * 100)
                base_y = y_pos + (row * item_height)
                
                btn_rect = (base_x - 10, base_y - 15, base_x + 80, base_y + 10)
                self.ui_layout['yolo_task_options'][task] = btn_rect

                radius = 6
                center_x, center_y = base_x, base_y - 5
                cv2.circle(canvas, (center_x, center_y), radius, (220, 220, 220), 1)
                if is_selected:
                    cv2.circle(canvas, (center_x, center_y), radius - 2, color, -1)
                cv2.putText(canvas, task, (center_x + 10, base_y), font, 0.5, color, 1)
            y_pos += ((len(tasks) - 1) // 2 + 1) * item_height
            y_pos += 15

            # Version selection
            cv2.putText(canvas, "Version:", (x_start + 10, y_pos), font, 0.7, (255, 255, 255), 1)
            y_pos += 30
            versions = YOLO_VERSIONS
            item_height = 30
            num_columns = 3
            for i, version in enumerate(versions):
                is_selected = version == self.yolo_version
                color = (0, 255, 0) if is_selected else (220, 220, 220)
                
                col = i % num_columns
                row = i // num_columns
                base_x = x_start + 40 + (col * 60)
                base_y = y_pos + (row * item_height)

                btn_rect = (base_x - 20, base_y - 15, base_x + 30, base_y + 10)
                self.ui_layout['yolo_version_options'][version] = btn_rect
                
                cv2.putText(canvas, version, (base_x - 10, base_y), font, 0.6, color, 1)
            
            y_pos += ((len(versions) - 1) // num_columns + 1) * item_height
            y_pos += 15

            # Size selection
            cv2.putText(canvas, "Size:", (x_start + 10, y_pos), font, 0.7, (255, 255, 255), 1)
            y_pos += 30
            sizes = self.yolo_valid_sizes.get(self.yolo_version, [])
            item_height = 30
            num_columns = 4 # Use 4 columns for sizes as they are short
            for i, size in enumerate(sizes):
                is_selected = size == self.yolo_size
                color = (0, 255, 0) if is_selected else (220, 220, 220)
                
                col = i % num_columns
                row = i // num_columns
                base_x = x_start + 30 + (col * 50)
                base_y = y_pos + (row * item_height)

                btn_rect = (base_x - 10, base_y - 15, base_x + 25, base_y + 10)
                self.ui_layout['yolo_size_options'][size] = btn_rect
                cv2.putText(canvas, size, (base_x, base_y), font, 0.6, color, 1)
            
            y_pos += ((len(sizes) - 1) // num_columns + 1) * item_height

    def run(self):
        if not self._setup_analyzer(): 
            return
        try:
            self._setup_video_source()
        except IOError: return

        self.controller.start_processing()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._update_yolo_trackbar()

        while self.running:
            if not self.controller.process_next_frame():
                break

            # --- Update Window Title ---
            model_display_name = self.args.model
            if self.args.model == "YOLO":
                task_str = f"-{self.yolo_task}" if self.yolo_task != "detect" else ""
                model_display_name = f"YOLO{self.yolo_version}{self.yolo_size}{task_str}"
            source_name = "Webcam" if self.controller.is_live else os.path.basename(self.args.video) if self.args.video else "Video File"
            title = f"AI Video Analysis - {model_display_name} on {source_name}"
            cv2.setWindowTitle(self.window_name, title)

            raw_bgr_frame = self.controller.latest_frame
            if raw_bgr_frame is None: continue

            annotated_bgr_frame = self.controller.latest_annotated_frame
            frame_height, frame_width, _ = raw_bgr_frame.shape

            # --- Define layout dimensions ---
            top_padding = 40
            bottom_padding = 220
            video_area_width = frame_width * 2
            total_width = video_area_width + self.sidebar_width
            total_height = top_padding + frame_height + bottom_padding

            # Create the main canvas
            main_canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

            # --- Draw titles ---
            cv2.putText(main_canvas, "Live Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(main_canvas, "Processed Feed", (frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # --- Place video frames ---
            main_canvas[top_padding:top_padding + frame_height, 0:frame_width] = raw_bgr_frame
            if annotated_bgr_frame is not None:
                processed_display = annotated_bgr_frame.copy()
            else:
                processed_display = np.zeros_like(raw_bgr_frame)
                cv2.putText(processed_display, "Awaiting analysis...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            
            state_text = f"State: {self.controller.processing_state.upper()}"
            cv2.putText(processed_display, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            main_canvas[top_padding:top_padding + frame_height, frame_width:video_area_width] = processed_display

            # --- Draw Sidebar ---
            self._draw_sidebar(main_canvas, video_area_width, total_height)

            # --- Draw JSON output area ---
            json_area_y_start = top_padding + frame_height
            json_area_width = video_area_width
            cv2.line(main_canvas, (0, json_area_y_start), (json_area_width, json_area_y_start), (100, 100, 100), 1)
            cv2.putText(main_canvas, "Analysis Results (Scroll with Up/Down arrows)", (10, json_area_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if self.controller.latest_analysis:
                try:
                    json_text = json.dumps(self.controller.latest_analysis, indent=2)
                    json_lines = json_text.split('\n')
                    text_canvas_height = len(json_lines) * self.json_line_height + (2 * self.json_padding)
                    text_canvas = np.zeros((text_canvas_height, json_area_width, 3), dtype=np.uint8)

                    for i, line in enumerate(json_lines):
                        y = (i * self.json_line_height) + self.json_line_height + self.json_padding
                        cv2.putText(text_canvas, line, (self.json_padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    visible_area_y_start = json_area_y_start + 40
                    visible_height = total_height - visible_area_y_start
                    max_scroll = max(0, text_canvas_height - visible_height)
                    self.json_scroll_offset_y = max(0, min(self.json_scroll_offset_y, max_scroll))

                    visible_text_area = text_canvas[self.json_scroll_offset_y : self.json_scroll_offset_y + visible_height, :]
                    main_canvas[visible_area_y_start : visible_area_y_start + visible_text_area.shape[0], 0:json_area_width] = visible_text_area
                except Exception as e:
                    logging.error(f"Failed to render JSON: {e}")

            cv2.imshow(self.window_name, main_canvas)

            # --- Handle Keyboard Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p'):
                self._handle_button_click('Start/Pause/Resume')
            elif key == 82: # Up arrow for scrolling JSON
                self.json_scroll_offset_y -= self.json_line_height * 2
            elif key == 84: # Down arrow for scrolling JSON
                self.json_scroll_offset_y += self.json_line_height * 2

        # --- Cleanup ---
        self.controller.stop_processing()
        cv2.destroyAllWindows()
        logging.info("Standalone application finished.")

if __name__ == "__main__":
    # The create_argument_parser function from core.py now handles loading .env
    # and defining all the common command-line arguments.
    parser = create_argument_parser()
    args = parser.parse_args()

    # The validation logic is also moved to core.py
    validate_yolo_args(parser, args)

    app = StandaloneApp(args)
    app.run()