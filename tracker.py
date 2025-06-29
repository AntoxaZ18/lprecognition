from trackers import BYTETracker
from argparse import Namespace
import numpy as np

class TrackerInput():
    '''class for feeding inference result to tracker'''
    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls_: np.ndarray):
        '''conf - probabilities
           xywh - coordinates in xywh format
           cls_ - class label 
        '''    
        self.conf = conf
        self.xywh = xywh
        self.cls = cls_

class ByteTracker():
    '''
    wrapper for bytetrack
    '''
    def __init__(self,
                 track_buffer=30,
                 track_high_thresh=0.6,
                 track_low_thresh=0.2,
                 fuse_score=0.6,
                 match_thresh=0.8,
                 new_track_thresh=0.5,
                 frame_rate=10):
        '''
        track_buffer - track buffer length
        track_high_thresh - threshold for the first association
        track_low_thresh - threshold for the second association
        fuse_score - Whether to fuse confidence scores with the iou distances before matching
        match_thresh - Matching threshold betwen track and detection.
        new_track_thresh - minimal threshold for starting tracking
        frame_rate - farme rate for object speed estimation
        '''
        track_params = Namespace(
            track_buffer=track_buffer,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            fuse_score=fuse_score,
            match_thresh=match_thresh,
            new_track_thresh=new_track_thresh
        )

        self.tracker = BYTETracker(track_params, frame_rate=frame_rate)

    def update(self, model_dets: TrackerInput, image_size: tuple[int, int]):
        '''
        call this method every frame inference
        return np.array of object tracking boxes in format 
        x, y, width, height, track_id, score, cls, idx 
        '''
        return self.tracker.update(model_dets, image_size)