import numpy as np
import helpers
import cv2

"""
  Abstract morphing algorithm class. Each deriving class should implement the
  @morph method, taking two images and number of steps and returning 
  a list of @n_steps images with increasing morphing.
"""
class MorphingAlgorithm(object):
  def morph(self, img_src, img_dst, n_steps = 10):
    raise NotImplementedError("Please Implement this method")
 

"""
  Simplest possible morpher, using linear interpolation.
"""
class SimpleLinearMorpher(MorphingAlgorithm):
  def morph(self, img_src, img_dst, n_steps = 10):
    from helpers import linear_interpolation
    return linear_interpolation(img_src, img_dst, n_steps)

def list_to_triangle(t):
  return np.array([(t[0], t[1]), (t[2], t[3]), (t[4], t[5])], dtype=np.float32)

"""
  TODO: implement this class
"""
class AdvancedMorphingAlgorithm(MorphingAlgorithm):
  # http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
  def morph(self, img_src, img_dst, n_steps = 10):
    if img_src.shape != img_dst.shape:
      raise ValueError("Source and destination images are not of the same size")

    height, width = img_src.shape[:-1]
    rect = (0, 0, width, height)
    
    """
    # 1) find corresponding points
    """
    pts_src = helpers.find_facial_landmarks(img_src)
    pts_dst = helpers.find_facial_landmarks(img_dst)
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
      
    triangles_src = subdiv_src.getTriangleList()
    
    # some triangles end up out of bounds, filter them out
    def check_fun(p):
      p1x, p1y, p2x, p2y, p3x, p3y = p
      return helpers.rect_contains(rect, [p1x, p1y]) and helpers.rect_contains(rect, [p2x, p2y]) and helpers.rect_contains(rect, [p3x, p3y])
    triangles_src = filter(check_fun, triangles_src)
    
    # create destination image trianguation
    triangles_dst = []
    for t in triangles_src:
      i1 = pts_src.index((t[0], t[1]))
      i2 = pts_src.index((t[2], t[3]))
      i3 = pts_src.index((t[4], t[5]))
      triangles_dst.append([pts_dst[i1][0], pts_dst[i1][1], pts_dst[i2][0], pts_dst[i2][1], pts_dst[i3][0], pts_dst[i3][1]])
    
    triangles_dst = np.array(triangles_dst, dtype = np.float32)
    
    """
    # 3) calculate the intermediate morphing steps of the triangles
    """
    triangles_morphed = []
    white = (255, 255, 255)
    output = []
    for i in xrange(0, n_steps):
      # 3a) add weigthed landmark points
      a = float(i) / (n_steps - 1)
      curr = np.array([(1 - a) * triangle_src + a * triangle_dst for (triangle_src, triangle_dst) in \
        zip(triangles_src, triangles_dst)])

      triangles_morphed.append(curr)

      # helpers.debug_triangles(curr, rect)
      
      # 3b) add weighted regions
      # TODO: this part is really really slow, integers should be used for images instead of floats for speed
      # TODO: figure out how to use integers
      img_dbg = np.zeros(shape = img_src.shape, dtype = img_src.dtype)
      for (ts, tc, td) in zip(triangles_src, curr, triangles_dst):
        # http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html#fillconvexpoly
        triangle_cur = list_to_triangle(tc)
        triangle_src = list_to_triangle(ts)
        triangle_dst = list_to_triangle(td)
        
        img_mask_src = np.zeros(img_src.shape, dtype=np.uint8)
        img_mask_dst = np.zeros(img_src.shape, dtype=np.uint8)
        
        cv2.fillConvexPoly(img_mask_src, np.array(triangle_src, dtype = np.int32), white)
        cv2.fillConvexPoly(img_mask_dst, np.array(triangle_dst, dtype = np.int32), white)
        
        # TODO: fix seams when blending, using integers will probably fix that
        img_masked_src = cv2.bitwise_and(img_src, img_mask_src)
        img_masked_dst = cv2.bitwise_and(img_dst, img_mask_dst)
        
        mat_src_cur = cv2.getAffineTransform(triangle_src, triangle_cur)
        mat_dst_cur = cv2.getAffineTransform(triangle_dst, triangle_cur)
        
        img_dbg = img_dbg + cv2.addWeighted(
          cv2.warpAffine(img_masked_src, mat_src_cur, (width, height)), 1 - a,
          cv2.warpAffine(img_masked_dst, mat_dst_cur, (width, height)), a, 0
          )                
          
      output.append(img_dbg)
      
      print "Morphing", a * 100, "% done."
      
      """    
      cv2.imshow("mask", img_dbg)
      cv2.waitKey(0)
      """
    
    helpers.save_images(output)


  
