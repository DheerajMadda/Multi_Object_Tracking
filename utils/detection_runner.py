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

class DetectionRunner:
    """
    This class runs the detection over the input image(s)/ video/ webcam
    """

    def __init__(self, detector):
        """
        This is the initialization method

        Parameters
        ----------
        detector : object
            A detection model object

        """
        self.detector = detector
        self.frame_resolution_wh = detector.frame_resolution_wh
        self.img_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

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
        
        while True:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_resolution_wh is not None:
                frame = cv2.resize(frame, self.frame_resolution_wh)

            width, height = frame.shape[:-1][::-1]

            if save_output:
                if is_sequence and vid_writer is None:
                    vid_writer = cv2.VideoWriter(save_output, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (width, height))

            results = self.detector(frame)
            if results:
                frame, _ = self.detector.draw_result(results[0], frame)

            end_time = time.perf_counter()

            # display fps
            fps = 1/ np.round(end_time - start_time, 2)
            pos = (20, 20)
            cv2.rectangle(frame, (0, 0), (100, 30), (0, 0, 0), -1)
            cv2.putText(frame, f'FPS: {int(fps)}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if display:
                cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detection', width=width, height=height)
                cv2.imshow('Detection', frame)

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
