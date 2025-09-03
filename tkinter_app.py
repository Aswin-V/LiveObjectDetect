import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import logging
import json
import os
from PIL import Image, ImageTk

from core import (AppController, create_argument_parser, validate_yolo_args,
                  YOLO_VERSIONS, YOLO_VALID_SIZES_BY_VERSION, YOLO_VALID_TASKS_BY_VERSION, create_yolo_analyzer_params)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TkinterApp(tk.Tk):
    _YOLO_VALID_SIZES = YOLO_VALID_SIZES_BY_VERSION
    _YOLO_VALID_TASKS = YOLO_VALID_TASKS_BY_VERSION

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("AI Video Analysis - Tkinter GUI")
        self.geometry("1300x800")

        self.controller = AppController()
        self.after_id = None
        self.selected_file_path = None

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_area = ttk.Frame(self.main_frame)
        self.video_area.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.video_area.grid_rowconfigure(1, weight=1)
        self.video_area.grid_columnconfigure(0, weight=1)
        self.video_area.grid_columnconfigure(1, weight=1)

        self.sidebar_frame = ttk.Frame(self.main_frame, width=280, padding="10")
        self.sidebar_frame.grid(row=0, column=1, sticky="ns")
        self.sidebar_frame.pack_propagate(False)

        # --- Video Display Widgets ---
        ttk.Label(self.video_area, text="Live Feed").grid(row=0, column=0, pady=(0, 5))
        self.raw_video_label = ttk.Label(self.video_area)
        self.raw_video_label.grid(row=1, column=0, sticky="nsew", padx=(0, 5))

        self.raw_frame_label = ttk.Label(self.video_area, text="Frame: -")
        self.raw_frame_label.grid(row=2, column=0, pady=(5, 0))

        ttk.Label(self.video_area, text="Processed Feed").grid(row=0, column=1, pady=(0, 5))
        self.processed_video_label = ttk.Label(self.video_area)
        self.processed_video_label.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        self.processed_frame_label = ttk.Label(self.video_area, text="Analyzed Frame: -")
        self.processed_frame_label.grid(row=2, column=1, pady=(5, 0))

        # --- JSON Display ---
        self.json_frame = ttk.Frame(self.video_area, padding="5")
        self.json_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.video_area.grid_rowconfigure(1, weight=1)

        self.json_frame.grid_rowconfigure(1, weight=1)
        self.json_frame.grid_columnconfigure(0, weight=1)
        ttk.Label(self.json_frame, text="Analysis Results:").pack(anchor='w')
        
        text_container = ttk.Frame(self.json_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        self.json_text = tk.Text(text_container, wrap=tk.WORD, height=10)
        self.json_scroll = ttk.Scrollbar(text_container, command=self.json_text.yview)
        self.json_text.config(yscrollcommand=self.json_scroll.set)
        self.json_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.json_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Sidebar Widgets ---
        self._create_sidebar_widgets()

        # --- Setup ---
        # Defer video source setup until user interaction
        if not self._setup_analyzer():
            self.destroy()
            return
        
        # If a video is passed via command line, set it as the initial selection
        if self.args.video:
            self.source_var.set("Video File")
            self.selected_file_path = self.args.video
            self._handle_source_selection()
            self.selected_file_label.config(text=os.path.basename(self.args.video))

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_sidebar_widgets(self):
        # --- Input Source Selection ---
        source_frame = ttk.LabelFrame(self.sidebar_frame, text="Input Source", padding="10")
        source_frame.pack(fill=tk.X, pady=10, ipady=5)

        self.source_var = tk.StringVar(value="Webcam")
        sources = ["Webcam", "Video File", "Image File"]
        for source in sources:
            rb = ttk.Radiobutton(source_frame, text=source, variable=self.source_var, value=source, command=self._handle_source_selection)
            rb.pack(anchor='w')

        self.browse_button = ttk.Button(source_frame, text="Browse...", command=self._browse_file, state=tk.DISABLED)
        self.browse_button.pack(fill=tk.X, pady=5)
        
        self.selected_file_label = ttk.Label(source_frame, text="No file selected", wraplength=240)
        self.selected_file_label.pack(anchor='w', pady=2)

        # --- Processing Controls ---
        controls_frame = ttk.LabelFrame(self.sidebar_frame, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5, ipady=5)
        
        self.start_stop_button = ttk.Button(controls_frame, text="Start Webcam", command=self._handle_start_stop)
        self.start_stop_button.pack(fill=tk.X, pady=5)
        
        self.pause_resume_button = ttk.Button(controls_frame, text="Pause", command=self._toggle_pause_resume, state=tk.DISABLED)
        self.pause_resume_button.pack(fill=tk.X, pady=5)

        # --- Model Selection ---
        model_frame = ttk.LabelFrame(self.sidebar_frame, text="Model", padding="10")
        model_frame.pack(fill=tk.X, pady=10, ipady=5)

        self.model_var = tk.StringVar(value=self.args.model)
        
        models = ["YOLO", "DeepFace", "Gemini"]
        for model in models:
            rb = ttk.Radiobutton(model_frame, text=model, variable=self.model_var, value=model, command=self._handle_model_selection)
            rb.pack(anchor='w')

        # --- YOLO Options ---
        self.yolo_frame = ttk.LabelFrame(self.sidebar_frame, text="YOLO Options", padding="10")
        self.yolo_frame.pack(fill=tk.X, pady=10, ipady=5)

        # Task Selection
        ttk.Label(self.yolo_frame, text="Task:").pack(anchor='w')
        self.yolo_task_var = tk.StringVar(value=self.args.yolo_task)
        yolo_tasks = self._YOLO_VALID_TASKS.get(self.args.yolo_version, [])
        self.yolo_task_menu = ttk.OptionMenu(self.yolo_frame, self.yolo_task_var, self.args.yolo_task, *yolo_tasks, command=self._handle_model_selection)
        self.yolo_task_menu.pack(fill=tk.X, pady=2)

        # Version
        ttk.Label(self.yolo_frame, text="Version:").pack(anchor='w')
        self.yolo_version_var = tk.StringVar(value=self.args.yolo_version)
        self.yolo_version_menu = ttk.OptionMenu(self.yolo_frame, self.yolo_version_var, self.args.yolo_version, *YOLO_VERSIONS, command=self._on_yolo_version_change)
        self.yolo_version_menu.pack(fill=tk.X, pady=2)

        # Size
        ttk.Label(self.yolo_frame, text="Size:").pack(anchor='w')
        self.yolo_size_var = tk.StringVar(value=self.args.yolo_size)
        self.yolo_size_menu = ttk.OptionMenu(self.yolo_frame, self.yolo_size_var, self.args.yolo_size, *self._YOLO_VALID_SIZES.get(self.args.yolo_version, []), command=self._handle_model_selection)
        self.yolo_size_menu.pack(fill=tk.X, pady=2)

        # Confidence
        ttk.Label(self.yolo_frame, text="Confidence:").pack(anchor='w')
        self.confidence_var = tk.DoubleVar(value=self.args.confidence)
        self.confidence_scale = ttk.Scale(self.yolo_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL, command=self._on_confidence_change)
        self.confidence_scale.pack(fill=tk.X, pady=2)
        
        self._toggle_yolo_options()

    def _toggle_yolo_options(self):
        state = 'normal' if self.model_var.get() == "YOLO" else 'disabled'
        for child in self.yolo_frame.winfo_children():
            child.configure(state=state)

    def _handle_source_selection(self):
        source = self.source_var.get()
        self.selected_file_path = None
        self.selected_file_label.config(text="No file selected")

        if source == "Webcam":
            self.browse_button.config(state=tk.DISABLED)
            self.start_stop_button.config(text="Start Webcam", state=tk.NORMAL)
        elif source == "Video File":
            self.browse_button.config(state=tk.NORMAL)
            self.start_stop_button.config(text="Start Video", state=tk.DISABLED)
        elif source == "Image File":
            self.browse_button.config(state=tk.NORMAL)
            self.start_stop_button.config(text="Process Image", state=tk.DISABLED)
        
        if self.controller.video_capture:
            self._stop_processing_loop()

    def _browse_file(self):
        source = self.source_var.get()
        if source == "Video File":
            filetypes = [("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")]
        elif source == "Image File":
            filetypes = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        else:
            return

        filepath = filedialog.askopenfilename(title=f"Select {source}", filetypes=filetypes)
        if filepath:
            self.selected_file_path = filepath
            self.selected_file_label.config(text=os.path.basename(filepath))
            self.start_stop_button.config(state=tk.NORMAL)
        else:
            self.selected_file_path = None
            self.selected_file_label.config(text="No file selected")
            self.start_stop_button.config(state=tk.DISABLED)

    def _handle_start_stop(self):
        if self.controller.video_capture:
            self._stop_processing_loop()
        else:
            source = self.source_var.get()
            if source == "Webcam":
                self._start_video_processing(None)
            elif source == "Video File" and self.selected_file_path:
                self._start_video_processing(self.selected_file_path)
            elif source == "Image File" and self.selected_file_path:
                self._process_single_image(self.selected_file_path)

    def _setup_analyzer(self):
        return self._handle_model_selection()

    def _handle_model_selection(self, _=None):
        model_name = self.model_var.get()
        analyzer_params = {}

        if model_name == "YOLO":
            analyzer_params = create_yolo_analyzer_params(
                version=self.yolo_version_var.get(),
                size=self.yolo_size_var.get(),
                task=self.yolo_task_var.get(),
                confidence=self.confidence_var.get()
            )
        elif model_name == "Gemini":
            analyzer_params = {"api_key": self.args.api_key}
        
        logging.info(f"Setting analyzer to {model_name} with params {analyzer_params}")
        analyzer_ready = self.controller.set_analyzer(model_name, **analyzer_params)
        
        if not analyzer_ready:
            error_msg = f"Failed to initialize {model_name} analyzer."
            if model_name == "Gemini":
                error_msg += "\nPlease provide an API key."
            logging.error(error_msg)
            messagebox.showerror("Analyzer Error", error_msg)
            return False
        
        self._toggle_yolo_options()
        return True

    def _on_yolo_version_change(self, _=None):
        """Handles YOLO version changes, updating dependent task and size menus."""
        new_version = self.yolo_version_var.get()

        # Update tasks menu
        valid_tasks = self._YOLO_VALID_TASKS.get(new_version, [])
        if self.yolo_task_var.get() not in valid_tasks:
            self.yolo_task_var.set(valid_tasks[0])
        
        task_menu = self.yolo_task_menu["menu"]
        task_menu.delete(0, "end")
        for task in valid_tasks:
            task_menu.add_command(label=task, command=lambda value=task: self.yolo_task_var.set(value))

        # Update sizes menu
        valid_sizes = self._YOLO_VALID_SIZES.get(new_version, [])
        if self.yolo_size_var.get() not in valid_sizes:
            self.yolo_size_var.set(valid_sizes[0])
        
        size_menu = self.yolo_size_menu["menu"]
        size_menu.delete(0, "end")
        for size in valid_sizes:
            size_menu.add_command(label=size, command=lambda value=size: self.yolo_size_var.set(value))

        self._handle_model_selection()

    def _on_confidence_change(self, value):
        self.controller.set_confidence_threshold(float(value))

    def _toggle_pause_resume(self):
        if self.controller.processing_state == 'running':
            self.controller.pause_processing()
        else:
            self.controller.start_processing()

    def _start_video_processing(self, video_path):
        """Starts a video stream (webcam or file) and the update loop."""
        try:
            if video_path:
                self.controller.start_video_file(video_path)
            else:
                self.controller.start_webcam()
        except IOError as e:
            logging.error(e)
            messagebox.showerror("Video Error", str(e))
            return

        self.controller.start_processing()
        self._update_frame()
        
        self.start_stop_button.config(text="Stop")
        self.pause_resume_button.config(state=tk.NORMAL)

    def _stop_processing_loop(self):
        """Stops the video stream and the update loop."""
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.controller.stop_processing()
        
        self.raw_video_label.config(image='')
        self.processed_video_label.config(image='')
        self.raw_photo = None
        self.processed_photo = None

        self.raw_frame_label.config(text="Frame: -")
        self.processed_frame_label.config(text="Analyzed Frame: -")

        self.pause_resume_button.config(state=tk.DISABLED)
        self._handle_source_selection()

    def _process_single_image(self, image_path):
        """Processes a single image and displays the result."""
        if self.controller.video_capture:
            self._stop_processing_loop()

        logging.info(f"Processing image: {image_path}")
        try:
            raw_bgr_frame, annotated_bgr_frame, analysis = self.controller.process_single_image(image_path)
            
            if "error" in analysis:
                messagebox.showerror("Analysis Error", analysis["error"])
                return

            # Convert BGR frames from controller to RGB for display
            raw_rgb_frame = cv2.cvtColor(raw_bgr_frame, cv2.COLOR_BGR2RGB)
            annotated_rgb_frame = None
            if annotated_bgr_frame is not None:
                annotated_rgb_frame = cv2.cvtColor(annotated_bgr_frame, cv2.COLOR_BGR2RGB)

            self._display_frames(raw_rgb_frame, annotated_rgb_frame)

            self.raw_frame_label.config(text="Frame: 1")
            self.processed_frame_label.config(text="Analyzed Frame: 1")

            if analysis:
                json_text = json.dumps(analysis, indent=2)
                self.json_text.delete('1.0', tk.END)
                self.json_text.insert(tk.END, json_text)

        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            messagebox.showerror("Image Error", f"Failed to process image: {e}")

    def _update_frame(self):
        if not self.controller.video_capture:
            self._stop_processing_loop()
            return

        if not self.controller.process_next_frame():
            self._stop_processing_loop()
            return

        raw_frame, annotated_frame = self.controller.get_display_frames()

        if raw_frame is not None:
            self._display_frames(raw_frame, annotated_frame) # These are already RGB

            self.raw_frame_label.config(text=f"Frame: {self.controller.frame_count}")
            if self.controller.analyzed_frame_number > 0:
                self.processed_frame_label.config(text=f"Analyzed Frame: {self.controller.analyzed_frame_number}")

        # Update JSON
        if self.controller.latest_analysis:
            try:
                json_text = json.dumps(self.controller.latest_analysis, indent=2)
                self.json_text.delete('1.0', tk.END)
                self.json_text.insert(tk.END, json_text)
            except (TypeError, ValueError):
                pass

        # Update button text
        state = self.controller.processing_state
        btn_text = "Pause" if state == 'running' else "Resume"
        self.pause_resume_button.config(text=btn_text)

        self.after_id = self.after(30, self._update_frame)

    def _display_frames(self, raw_rgb_frame, annotated_rgb_frame):
        """Helper to resize RGB frames and display them in the UI."""
        if raw_rgb_frame is None:
            return

        h, w, _ = raw_rgb_frame.shape
        max_w = (self.winfo_width() - self.sidebar_frame.winfo_width()) // 2 - 40
        if max_w <= 0: return

        scale = max_w / w
        new_w, new_h = int(w * scale), int(h * scale)

        img_raw = Image.fromarray(raw_rgb_frame)
        img_raw = img_raw.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.raw_photo = ImageTk.PhotoImage(image=img_raw)
        self.raw_video_label.config(image=self.raw_photo)

        if annotated_rgb_frame is not None:
            img_processed = Image.fromarray(annotated_rgb_frame)
            img_processed = img_processed.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.processed_photo = ImageTk.PhotoImage(image=img_processed)
            self.processed_video_label.config(image=self.processed_photo)

    def _on_closing(self):
        self._stop_processing_loop()
        self.destroy()

def main():
    # The create_argument_parser function from core.py now handles loading .env
    # and defining all the common command-line arguments.
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # The validation logic is also moved to core.py
    validate_yolo_args(parser, args)

    app = TkinterApp(args)
    app.mainloop()

if __name__ == "__main__":
    main()