import os
from glob import glob
import cv2
import time
import numpy as np

class ImageCapture:
    """
    This class reads the images or sequence of images (given a directory)
    """
    def __init__(self, path, is_file):
        """
        This is the initialization method

        Parameters
        ----------
        path : str
            A path to an image file or a directory containing images
        is_file : bool
            A boolean representing whether the path provided is a file or not

        """
        if is_file:
            self.images = [path]
        else:
            self.images = sorted(glob(os.path.join(path, "*")))
        self.index = 0

    def read(self):
        """
        This method reads the image

        Returns
        -------
        tuple
        
        """
        try:
            img = cv2.imread(self.images[self.index], cv2.IMREAD_COLOR)
            self.index += 1
            return True, img

        except IndexError:
            return False, None

    def release(self):
        """
        This method deletes the instance

        Returns
        -------
        None
        
        """
        del self

class TrackingRunner:
    """
    This class runs the detection + tracking over the input image(s)/ video/ webcam
    """

    def __init__(self, detector, tracker):
        """
        This is the initialization method

        Parameters
        ----------
        detector : object
            A detection model object
        tracker : object
            A tracking object

        """

        self.detector = detector
        self.tracker = tracker
        self.frame_resolution_wh = detector.frame_resolution_wh
        self.img_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        self.max_store_prev_track_ids = 500

    def warmup_tracker(self):
        """
        This method warmups the tracker if it uses a model for reidentification

        Returns
        -------
        None

        """
        # if tracker is using model, then perform warmup
        if hasattr(self.tracker, 'model'):
            if hasattr(self.tracker.model, 'warmup'):
                self.tracker.model.warmup()

    def update_track_count(self, detections_list):
        """
        This method is used to update the track count

        Parameters
        ----------
        detections_list : list
            A list containing the detections
            
        Returns
        -------
        None
        
        """
        for detections in detections_list:
            tracks = detections.tracker_id.tolist()
            tracks = list(set(tracks).difference(set(self.store_prev_track_ids)))
            self.tracks_count += len(tracks)
            self.store_prev_track_ids += tracks
            self.store_prev_track_ids = self.store_prev_track_ids[-self.max_store_prev_track_ids:]
                 
    def run(self, source, save_output=None, display=False, keyboard_interrupt_key="q"):
        """
        This method runs the detection over the input image(s)/ video/ webcam

        Parameters
        ----------
        source : [int, str]
            A source input to a webcam index or an image path, directory path, or a video path
        save_output : [None, str]
            Saves the output of the detection to a given file
        display : bool
            A boolean defining whether to display the detection output frame in the window or not
        keyboard_interrupt_key : str
            A string representing a keyboard button to exit the running detection process

        Returns
        -------
        object

        """
        self.store_prev_track_ids = []
        self.tracks_count = 0

        is_sequence = True
        vid_writer = None
        result = None

        if type(source) == int:
            # webcam
            cap = cv2.VideoCapture(source)
            assert cap.isOpened()

        elif os.path.isfile(source):
            if any([ext in source.lower() for ext in self.img_extensions]):
                # path is an image
                cap = ImageCapture(source, is_file=True)
                is_sequence = False
                save_output = None
            else:
                # path is a video
                cap = cv2.VideoCapture(source)
                assert cap.isOpened()
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        elif os.path.isdir(source):
            # path is a directory
            cap = ImageCapture(source, is_file=False)
        else:
            raise Exception(f"'{source}' not found!")

        if is_sequence:
            print(f"Press the keyboard button '{keyboard_interrupt_key}' to exit.")
        
        # warmup tracker
        self.warmup_tracker()
        curr_frame, prev_frame = None, None

        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_resolution_wh is not None:
                frame = cv2.resize(frame, self.frame_resolution_wh)

            width, height = frame.shape[:-1][::-1]
            curr_frame = frame.copy()

            if save_output:
                if is_sequence and vid_writer is None:
                    vid_writer = cv2.VideoWriter(save_output, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

            results = self.detector(frame)
            
            # track
            if hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, 'camera_update'):
                if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                    self.tracker.tracker.camera_update(prev_frame, curr_frame)

            if results:
                result = results[0]
                tracker_result = self.tracker.update(result, frame)
                frame, detections_list = self.detector.draw_result(tracker_result, frame)
                self.update_track_count(detections_list)

            end_time = time.perf_counter()

            # display tracks count
            pos = (frame.shape[1] - 110, 20)
            cv2.rectangle(frame, (frame.shape[1] - 130, 0), (frame.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(frame, f'COUNT: {self.tracks_count}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # display fps
            fps = 1/ np.round(end_time - start_time, 2)
            pos = (20, 20)
            cv2.rectangle(frame, (0, 0), (100, 30), (0, 0, 0), -1)
            cv2.putText(frame, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            prev_frame = curr_frame

            if display:
                cv2.namedWindow('Detection-Tracking', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detection-Tracking', width=width, height=height)
                cv2.imshow('Detection-Tracking', frame)

            if is_sequence:
                key = cv2.waitKey(5)
                if key == ord(keyboard_interrupt_key):
                    break
            else:
                result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if display:
                    key = cv2.waitKey(0)
                    if key == ord(keyboard_interrupt_key):
                        break

            if save_output:
                vid_writer.write(frame)
        
        if save_output:
            vid_writer.release()
        cap.release()
        cv2.destroyAllWindows()

        return result
