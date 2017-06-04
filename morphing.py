import numpy as np
import helpers
import cv2


CURRENT_MORPH_WIN_NAME = "Current morph"
CURRENT_MORPH_SLIDER_NAME = "Intensity"
MORPH_LEVELS = 100.0

"""
  Abstract morphing algorithm class. Each deriving class should implement the
  @morph method, taking two images and number of steps and returning 
  a list of @n_steps images with increasing morphing.
"""
class MorphingAlgorithm(object):
  # morph a single image with desired intensity alpha
  def step(self, alpha):
    raise NotImplementedError("Please Implement this method")
      
  # return an array of n_steps progressively morphed images
  def morph(self, n_steps):
    output = []

    if n_steps != 0:
      speedFactor = 1.7
      for i in xrange(0, n_steps):
        # 3a) add weighted landmark points
        alpha = helpers.speedUpMorph(float(i) / (n_steps - 1), speedFactor)
        output.append(self.step(alpha))
        
    else:
      cv2.namedWindow(CURRENT_MORPH_WIN_NAME)
      alpha = cv2.getTrackbarPos(CURRENT_MORPH_SLIDER_NAME, CURRENT_MORPH_WIN_NAME) / MORPH_LEVELS
      while(True):
        cv2.createTrackbar(CURRENT_MORPH_SLIDER_NAME, CURRENT_MORPH_WIN_NAME, 0, int(MORPH_LEVELS), lambda x: None)
        cv2.setTrackbarPos(CURRENT_MORPH_SLIDER_NAME, CURRENT_MORPH_WIN_NAME, int(alpha * MORPH_LEVELS))
        
        out = self.step(alpha)
        
        cv2.imshow(CURRENT_MORPH_WIN_NAME, out)
        cv2.waitKey(1)
   
        alpha = cv2.getTrackbarPos(CURRENT_MORPH_SLIDER_NAME, CURRENT_MORPH_WIN_NAME) / MORPH_LEVELS

    return output
            
"""
  Simplest possible morpher, using linear interpolation.
"""
class SimpleLinearMorpher(MorphingAlgorithm):
  def __init__(self, img_src, img_dst):
    self.img_src = img_src
    self.img_dst = img_dst

  def step(self, alpha):
    return cv2.addWeighted(self.img_src, 1 - alpha, self.img_dst, alpha, 0)

def list_to_triangle(t):
  return np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], dtype=np.float32)

def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
# https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.py
def morph_triangle(img1, img2, img, t1, t2, t, alpha):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32(t1))
    r2 = cv2.boundingRect(np.float32(t2))
    r = cv2.boundingRect(np.float32(t))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

# http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
class AdvancedMorphingAlgorithm(MorphingAlgorithm):
  def __init__(self, img_src, img_dst):
    if img_src.shape != img_dst.shape:
      raise ValueError("Source and destination images are not of the same size")

    height, width = img_src.shape[:-1]
    rect = (0, 0, width, height)
    
    self.img_src = img_src
    self.img_dst = img_dst
    
    """
    # 1) find corresponding points
    """
    pts_src = helpers.find_facial_landmarks(img_src)
    pts_dst = helpers.find_facial_landmarks(img_dst)
    
    if pts_src is None or pts_dst is None:
      print "Advanced algorithm found no faces, try to supply better images"
      return None
    
    extra_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1), \
      (0, height / 2), (width - 1, height / 2), (width / 2, 0), (width / 2, height - 1)]
    pts_src.extend(extra_points)
    pts_dst.extend(extra_points)

    """    
    # 2) calculate Delaunay triangulation for source image and apply the same triangulation to the destination
    # http://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
    """
    # create an instance of Subdiv2D
    subdiv_src = cv2.Subdiv2D(rect)
    # insert the points into the subdiv object
    for ps in pts_src:
      subdiv_src.insert(ps)
      
    self.triangles_src = subdiv_src.getTriangleList()
    
    # some triangles end up out of bounds, filter them out
    def check_fun(p):
      p1x, p1y, p2x, p2y, p3x, p3y = p
      return helpers.rect_contains(rect, [p1x, p1y]) and helpers.rect_contains(rect, [p2x, p2y]) and helpers.rect_contains(rect, [p3x, p3y])
    self.triangles_src = filter(check_fun, self.triangles_src)
    
     # create destination image trianguation
    self.triangles_dst = []
    for t in self.triangles_src:
      i1 = pts_src.index((t[0], t[1]))
      i2 = pts_src.index((t[2], t[3]))
      i3 = pts_src.index((t[4], t[5]))
      self.triangles_dst.append([pts_dst[i1][0], pts_dst[i1][1], pts_dst[i2][0], pts_dst[i2][1], pts_dst[i3][0], pts_dst[i3][1]])
    
    self.triangles_dst = np.array(self.triangles_dst, dtype = np.float32)

  def step(self, alpha):
    # 3) calculate the intermediate morphing steps of the triangles
    triangles_morphed = []
    white = (255, 255, 255)

    curr = np.array([(1 - alpha) * triangle_src + alpha * triangle_dst for (triangle_src, triangle_dst) in \
      zip(self.triangles_src, self.triangles_dst)])

    triangles_morphed.append(curr)

    # 3b) add weighted regions
    # Allocate space for final output
    output = np.zeros(self.img_src.shape, dtype = self.img_src.dtype)

    for j in xrange(0, len(self.triangles_src)):
      morph_triangle(self.img_src, self.img_dst, output, \
       list_to_triangle(self.triangles_src[j]), list_to_triangle(self.triangles_dst[j]), list_to_triangle(curr[j].ravel()), alpha)

    return output

  
