import cv2
import numpy as np

def structured_edge_bgr(img_e):

    #img_e_orig = img_e.copy()
    image_e = cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB)
    image_e = image_e.astype(np.float32) / 255.0
    """
    gray_e = cv2.cvtColor(img_e_orig, cv2.COLOR_BGR2GRAY)
    blurred_e = cv2.GaussianBlur(gray_e, (5, 5), 0)
    """
    edge_detector_e = cv2.ximgproc.createStructuredEdgeDetection('StructuredEdgeModel/model.yml')
    edges_e = edge_detector_e.detectEdges(image_e)
    return edges_e