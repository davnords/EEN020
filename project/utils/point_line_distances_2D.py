import numpy as np 

def point_line_distance_2D(p, l):
    """Calculates the distance between a point and a line in 2D space."""
    if len(l) != 3:
        raise ValueError('Line must be represented by three coordinates')
    if len(p) != 2:
        raise ValueError('Point must be represented by two coordinates')
    a, b, c = l
    return np.abs(a*p[0] + b*p[1] + c)/np.linalg.norm(l[:2])
    
