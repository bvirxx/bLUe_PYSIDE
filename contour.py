import cv2
import numpy as np
"""
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }
"""
"""
Python: cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) → None
Parameters:
image – Destination image.
contours – All the input contours. Each contour is stored as a point vector.
contourIdx – Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
color – Color of the contours.
thickness – Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
lineType – Line connectivity. See line() for details.
hierarchy – Optional information about hierarchy. It is only needed if you want to draw only some of the contours (see maxLevel ).
maxLevel – Maximal level for drawn contours. If it is 0, only the specified contour is drawn. If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This parameter is only taken into account when there is hierarchy available.
offset – Optional contour shift parameter. Shift all the drawn contours by the specified  \texttt{offset}=(dx,dy) .
contour – Pointer to the first contour.
external_color – Color of external contours.
hole_color – Color of internal contours (holes).
"""

"""
Python: cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy
Parameters:
image – Source, an 8-bit single-channel image. Non-zero pixels are treated as 1’s. Zero pixels remain 0’s, so the image is treated as binary . You can use compare() , inRange() , threshold() , adaptiveThreshold() , Canny() , and others to create a binary image out of a grayscale or color one. The function modifies the image while extracting the contours. If mode equals to CV_RETR_CCOMP or CV_RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).
contours – Detected contours. Each contour is stored as a vector of points.
hierarchy – Optional output vector, containing information about the image topology. It has as many elements as the number of contours.
For each i-th contour contours[i] , the elements hierarchy[i][0] , hiearchy[i][1] , hiearchy[i][2] , and hiearchy[i][3] are set
to 0-based indices in contours of the next and previous contours at the same hierarchical level, the first child contour
and the parent contour, respectively. If for the contour i there are no next, previous, parent, or nested contours,
the corresponding elements of hierarchy[i] will be negative.
mode –
Contour retrieval mode (if you use Python see also a note below).

CV_RETR_EXTERNAL retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
CV_RETR_LIST retrieves all of the contours without establishing any hierarchical relationships.
CV_RETR_CCOMP retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components. At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours. This full hierarchy is built and shown in the OpenCV contours.c demo.
method –
Contour approximation method (if you use Python see also a note below).

CV_CHAIN_APPROX_NONE stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS applies one of the flavors of the Teh-Chin chain approximation algorithm. See [TehChin89] for details.
offset – Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.
"""

"""
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges
Parameters:
image – single-channel 8-bit input image.
edges – output edge map; it has the same size and type as image .
threshold1 – first threshold for the hysteresis procedure.
threshold2 – second threshold for the hysteresis procedure.
apertureSize – aperture size for the Sobel() operator.
L2gradient – a flag, indicating whether a more accurate  L_2 norm  =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default  L_1 norm  =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
"""

blue, green, red = cv2.split(img)


def medianCanny(img, thresh1, thresh2): # 0.2 0.3
    median = np.median(img)
    img = cv2.Canny(img, int(thresh1 * median), int(thresh2 * median))
    return img

blue_edges = medianCanny(blue, 0.2, 0.3)
green_edges = medianCanny(green, 0.2, 0.3)
red_edges = medianCanny(red, 0.2, 0.3)

edges = blue_edges | green_edges | red_edges

contours, hierarchy=cv2.findContours( edges, cv2.CV_RETR_TREE, cv2.CV_CHAIN_APPROX_SIMPLE);

hierarchy = hierarchy[0] # get the actual (inner) list of hierarchy descriptions python returns [hierarchy]
for component in zip(contours, hierarchy):
    currentContour = component[0]
    currentHierarchy = component[1]
    x,y,w,h = cv2.boundingRect(currentContour)
    if currentHierarchy[2] < 0:
        # these are the innermost child components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    elif currentHierarchy[3] < 0:
        # these are the outermost parent components
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
