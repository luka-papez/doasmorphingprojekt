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
      
    # 1) find corresponding points
    pts_src = helpers.find_facial_landmarks(img_src)
    pts_dst = helpers.find_facial_landmarks(img_dst)
    extra_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1), \
      (0, height / 2), (width - 1, height / 2), (width / 2, 0), (width / 2, height - 1)]
    pts_src.extend(extra_points)
    pts_dst.extend(extra_points)
    
    # 2) calculate Delaunay triangulation
    triangles_src, triangles_dst = self.find_Delaunay_triangles(pts_src, pts_dst, rect, img_src)
    
    # 3) 
    output = [] # TODO region cropping and creating images
    
    triangles_morphed = []
    outs = [] # TODO remove debugging
    for i in xrange(0, n_steps):
      # 3a) add weigthed landmark points
      a = float(i) / (n_steps - 1)
      curr = np.array([(1 - a) * triangle_src + a * triangle_dst for (triangle_src, triangle_dst) in \
        zip(triangles_src, triangles_dst)])
      outs.append(helpers.debug_draw_triangles(curr, rect)) # TODO remove debugging
      triangles_morphed.append(curr)
    
    # TODO region cropping and color interpolation
    
    print triangles_morphed[0].shape
    helpers.save_images(outs) # TODO remove debugging
    
    

  # http://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/
  def find_Delaunay_triangles(self, pts_src, pts_dst, rect = None, img_src = None):
    # create an instance of Subdiv2D
    subdiv_src = cv2.Subdiv2D(rect);
    subdiv_dst = cv2.Subdiv2D(rect);
    
    # insert the points into the subdiv object
    for (ps, pd) in zip(pts_src, pts_dst):
      subdiv_src.insert(ps)
      subdiv_dst.insert(pd)
      
    triangles_src = subdiv_src.getTriangleList()
    triangles_dst = subdiv_dst.getTriangleList()
    
    # debugging: draw the found triangles
    if img_src is not None:
      helpers.debug_draw_triangles(triangles_src, rect, img_src)
        
    return triangles_src, triangles_dst

  
