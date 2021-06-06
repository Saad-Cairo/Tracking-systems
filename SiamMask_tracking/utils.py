from shapely.geometry import Polygon
from box import *

def box_polygon_iou(box, polygon):
    box_vertices = [[box.x1, box.y1], [box.x1, box.y2], [box.x2, box.y2], [box.x2, box.y1]]
    polygon1 = Polygon(box_vertices)
    polygon2 = Polygon(polygon)

    intersection = polygon1.intersection(polygon2).area
    union = polygon2.union(polygon1).area
    '''if polygon1.area != 0.0 and polygon2.area != 0.0 and (intersection / polygon1.area > 0.5 or intersection / polygon2.area > 0.5):
        return 1.0'''
    iou = intersection / union
    return iou
def box_box_iou(box1, box2):
    box1 = [[box1.x1, box1.y1], [box1.x1, box1.y2], [box1.x2, box1.y2], [box1.x2, box1.y1]]
    box2 = [[box2.x1, box2.y1], [box2.x1, box2.y2], [box2.x2, box2.y2], [box2.x2, box2.y1]]
    poly_1 = Polygon(box1)
    poly_2 = Polygon(box2)
    inter = poly_1.intersection(poly_2).area

    if poly_1.area != 0.0 and poly_2.area != 0.0 and (inter / poly_1.area > 0.5 or inter / poly_2.area > 0.5):
        return 1.0
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    return iou

def convert_polygon_to_box(polygon):
    top_left_point_x = min(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    top_left_point_y = min(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])

    bottom_right_point_x = max(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0])
    bottom_right_point_y = max(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1])

    return Box((top_left_point_x, top_left_point_y, bottom_right_point_x, bottom_right_point_y), None)

def from_ROI_to_Box(x, y, w, h):

    return x, y, x+w, y+h