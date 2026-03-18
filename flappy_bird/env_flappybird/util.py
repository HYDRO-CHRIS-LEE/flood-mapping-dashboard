import math


def distance(point_1=(0, 0), point_2=(0, 0)):
    """Returns the distance between two points"""
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width / 2
    image.anchor_y = image.height / 2

def value_in_range(v, min, max):
    return (v >= min) and (v <= max)

def check_overlap(p_x, p_y, p_r, w_x, w_y, w_w, w_h):

    x1, y1, x2, y2 = p_x, p_y, w_x, w_y
    w1, h1, w2, h2 = p_r, p_r, w_w, w_h
    x1 -= w1/2
    y1 -= h1/2
    x2 -= w2/2
    y2 -= h2/2

    overlap_x = value_in_range(x1, x2, x2 + w2) or value_in_range(x2, x1, x1 + w1)
    overlap_y = value_in_range(y1, y2, y2 + h2) or value_in_range(y2, y1, y1 + h1)

    return overlap_x and overlap_y
