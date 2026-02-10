import numpy as np 
import pandas as pd 
import shapely.affinity
from shapely.geometry import Polygon, Point

# Venman
P1 = Polygon([(-468,-82), (-468,44), (-314,44), (-305,12), (-192,44), (-192,-82)])
P2 = Polygon([(-78,-171), (-78,-215), (-305,-215), (-305,-171)])
P3 = Polygon([(-62, 70), (95, 70), (142, 0), (140, -142), (-62, -142)])

# P1 = Polygon([(-468,-82), (-468,44), (-192,44), (-192,-82)])
# P2 = Polygon([(-78,-171), (-78,-215), (-305,-215), (-305,-171)])
# P3 = Polygon([(183,-142), (183,70), (-62,70), (-62,-142)])

# Karawatha
P4 = Polygon([(-150, 8), (300,8), (300,-210), (-150,-210)])
P5 = Polygon([(-215,618), (-74,618), (-74,423), (-215,423)])
P6 = Polygon([(-513,300), (-513,37), (-321,37), (-321,300)])

def make_circle(x,y, radius = 30):
    circle = Point(x,y).buffer(1) 
    circle = shapely.affinity.scale(circle, radius, radius)
    return circle

# Venman
B1 = make_circle(-63, 40)
B2 = make_circle(114,-143)
B3 = make_circle(-77,-205)
B4 = make_circle(-310,-171)
B5 = make_circle(-433,-82)
B6 = make_circle(-189,12)

# Karawatha
B7 = make_circle(-216,606)
B8 = make_circle(-98,428)
B9 = make_circle(-316,260)
B10 = make_circle(-321,63)
B11 = make_circle(-149,-22)
B12 = make_circle(300,-134)



def check_in_test_set(easting, northing, test_polygons, exclude_polygons):
    split = 'train'
    point = Point(easting, northing)
    for poly in test_polygons:
        if poly.contains(point):
            split = 'test'
            return split 
    for poly in exclude_polygons:
        if poly.contains(point):
            split = 'buffer'
            return split 
    return split  


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray, pose = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
        self.pose = pose 