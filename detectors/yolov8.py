import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

class YoloV8Utils:
    """
    This class provides post inference utilities
    """

    def __init__(self):
        """
        This is the initialization method
        """

        self.zones = [None]
        self.zone_annotators = [None]
        self.colors = sv.ColorPalette.default()
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors, 
                thickness=2, 
                text_thickness=1, 
                text_scale=0.5,
                text_padding=2
            )
        ]
        self.frame_resolution_wh = None

    def detect_zones(self, zones=[], frame_resolution_wh=None):
        """
        This method is used to detect only in specified zones

        Parameters
        ----------
        zones : list
            A list containing the zone(s)
        frame_resolution_wh : [None, tuple]
            A None object or a tuple (widht, height) representing the width and height of the frame

        Returns
        -------
        None
        
        """

        if zones:
            self.zones = []
            self.zone_annotators = []
            self.box_annotators = []
            self.frame_resolution_wh = frame_resolution_wh

            for index, zone in enumerate(zones):
                self.zones.append(
                    sv.PolygonZone(
                        polygon=np.array(zone).astype(np.int32), 
                        frame_resolution_wh=self.frame_resolution_wh
                    )
                )
                self.zone_annotators.append(
                    sv.PolygonZoneAnnotator(
                        zone=self.zones[index], 
                        color=self.colors.by_idx(index), 
                        thickness=2,
                        text_thickness=1,
                        text_scale=0.5,
                        text_padding=2
                    )
                )
                self.box_annotators.append(
                    sv.BoxAnnotator(
                        color=self.colors.by_idx(index), 
                        thickness=2, 
                        text_thickness=1, 
                        text_scale=0.5,
                        text_padding=2
                    )
                )

    def get_detections(self, result):
        """
        This method is used to get the detections from the detection model inference result

        Parameters
        ----------
        result : object
            An object defining the detection model inference result
            
        Returns
        -------
        list
        
        """
        if isinstance(result, (list,)):
            result = np.array(result)

        if result.shape[-1] == 6:
            # detector result where result.shape[-1] is 6 and result is torch.tensor
            confidence_pos = 4
            class_id_pos = 5
            result = result.cpu().numpy()
            tracker_id = None
        else:
            # tracker result where result.shape[-1] is 7 and result[:, 4] is tracker_id
            track_id_pos = 4
            class_id_pos = 5
            confidence_pos = 6
            tracker_id = result[:, track_id_pos].astype(int)
            
        xyxy = result[:, 0:4]
        confidence = result[:, confidence_pos]
        class_id = result[:, class_id_pos].astype(int)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id
        )

        # Get detections by zones
        detections_list = []
        for zone in self.zones:
            if zone is not None:
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                detections_list.append(detections_filtered)
            else:
                # zone is not available
                detections_list.append(detections)

        return detections_list

    def annotate_frame(self, detections_list, frame):
        """
        This method is used to annotate the frame

        Parameters
        ----------
        detections_list : list
            A list containing the detections
        frame : object
            A numpy array object defining an image
            
        Returns
        -------
        object
        
        """

        for detections, zone, zone_annotator, box_annotator in \
            zip(detections_list, self.zones, self.zone_annotators, self.box_annotators):

            if zone is not None:
                frame = zone_annotator.annotate(scene=frame)

            if self.display_labels:
                # Format custom labels
                labels = [
                    f"{'' if tracker_id is None else f'#{tracker_id} '}{self.class_names_dict[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id in detections
                ]
                skip_label = False
            else:
                labels = None
                skip_label = True
                
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels, skip_label=skip_label)

        return frame

    def draw_result(self, result, frame):
        """
        This method is used to get the detections from the model inference result,
        annotate the frame and update the tracker

        Parameters
        ----------
        result : object
            An object defining the detection model inference result
        frame : object
            A numpy array object defining an image
            
        Returns
        -------
        tuple
        
        """
        if len(result) == 0:
            for zone, zone_annotator in zip(self.zones, self.zone_annotators):
                if zone is not None:
                    frame = zone_annotator.annotate(scene=frame)
            return frame, []

        # Get detections_list
        detections_list = self.get_detections(result)

        # Annotate frame
        frame = self.annotate_frame(detections_list, frame)

        return frame, detections_list

class YoloV8(YoloV8Utils):
    """
    This class loads the detection model and predicts the detections
    """

    def __init__(self, weights="yolov8n.pt", conf_thres=0.45, display_labels=True, device=torch.device("cpu")):
        """
        This is the initialization method

        Parameters
        ----------
        weights : str
            A path to the model weights
        conf_thres: int
            An integer defining the threshold value for detected class confidence
        display_labels : bool
            A boolean defining whether to display the labels in the output frame or not
        device : object
            A torch.device() object that defines the target device for the inference

        """

        super(YoloV8, self).__init__()

        self.model = self.load_model(weights, device)
        self.class_names_dict = self.model.model.names
        self.conf_thres = conf_thres
        self.display_labels = display_labels
        self.classes = None
    
    def load_model(self, weights, device):
        """
        This method loads the model

        Parameters
        ----------
        weights : str
            A path to the model weights
        device : object
            A torch.device() object that defines the target device for the inference

        Returns
        -------
        object
        
        """
        model = YOLO(weights)
        model.fuse()
        model.to(device)
        return model

    def detect_classes(self, classes=[]):
        """
        This method is used to detect only specified classes

        Parameters
        ----------
        classes : list
            A list containing the class labels

        Returns
        -------
        None
        
        """
        if classes:
            is_int = all([isinstance(_cls, int) for _cls in classes])
            assert is_int == True, "provided list must contain only integer values"
            self.classes = classes

    def get_result_by_classes(self, result):
        """
        This method is used to filter the result by classes

        Parameters
        ----------
        result : object
            A torch.tensor() object defining the detection model inference result
            
        Returns
        -------
        object
        
        """
       
        # Filter results by specified classes only
        class_id_pos = 5
        class_ids = result[:, class_id_pos].to(torch.int32)

        indices = [
                index 
                for index, _cls in enumerate(class_ids) 
                if _cls in self.classes
            ]

        if indices:
            return result[indices]
        else:
            return result

    def predict(self, frame):
        """
        This method performs inference

        Parameters
        ----------
        frame : object
            A numpy array object defining an image

        Returns
        -------
        list
        
        """
        outputs = self.model(frame)

        results = []
        confidence_pos = 4

        for output in outputs:
            result = output.boxes.data # (x1, y1, x2, y2, class_confidence, class_id)

            # Filter result by confidence threshold
            result = result[result[:, confidence_pos] > self.conf_thres]

            if self.classes is not None:
                # Filter result by classes
                result = self.get_result_by_classes(result)

            results.append(result)
            
        return results

    def __call__(self, frame):
        """
        This method performs inference

        Parameters
        ----------
        frame : object
            A numpy array object defining an image

        Returns
        -------
        list
        
        """
        return self.predict(frame)
