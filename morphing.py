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
    
    print rect
    print
      
    # 1) find corresponding points
    pts_src = helpers.find_facial_landmarks(img_src)
    pts_dst = helpers.find_facial_landmarks(img_dst)
    extra_points = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1), \
      (0, height / 2), (width - 1, height / 2), (width / 2, 0), (width / 2, height - 1)]
    pts_src.extend(extra_points)
    pts_dst.extend(extra_points)
    
    # 2) calculate Delaunay triangulation
    """
    To circumvent this problem, you can either create triangulation on just one image and use it for the other as well ( see which points are forming the triangle). Or you can take the mean of the two face points and do the triangulation once.
    """
    triangles_src, triangles_dst = self.find_Delaunay_triangles(pts_src, pts_dst, rect)
    # TODO: triangulacija nije jednoznacna, treba pronaci indekse tocaka u triangulaciji i onda poslozit iste indekse da budu
    
    print pts_src
    print
    
    
    np.clip(triangles_src, 0, height - 1, out=triangles_src)
    
    print triangles_src
    
    # TODO: neki trokutovi su totalno glupi, detektirati te!!
    
    for (ind, t) in enumerate(triangles_src):
      i1 = pts_src.index((t[0], t[1]))
      i2 = pts_src.index((t[2], t[3]))
      i3 = pts_src.index((t[4], t[5]))
      triangles_dst[ind] = [pts_dst[i1][0], pts_dst[i1][1], pts_dst[i2][0], pts_dst[i2][1], pts_dst[i3][0], pts_dst[i3][1]]
    
    # 3) 
    output = [] # TODO region cropping and creating images
    
    print triangles_src[4], triangles_dst[0]
    
    triangles_morphed = []
    outs = [] # TODO remove debugging
    for i in xrange(0, n_steps):
      # 3a) add weigthed landmark points
      a = float(i) / (n_steps - 1)
      curr = np.array([(1 - a) * triangle_src + a * triangle_dst for (triangle_src, triangle_dst) in \
        zip(triangles_src, triangles_dst)])
        
      some = 4
      
      print "Curr:", curr[some]
      print "Src: ", triangles_src[some]
      print "Dst: ", triangles_dst[some]
      print
      
      pt1 = (curr[some][0], curr[some][1])
      pt2 = (curr[some][2], curr[some][3])
      pt3 = (curr[some][4], curr[some][5])
      
      delaunay_color = (0, 0, 255)
    
      img_dbg = np.zeros((rect[3], rect[2], 3))
      

    
      outs.append(helpers.debug_draw_triangles(curr, rect, img_dbg)) # TODO remove debugging
            
      cv2.line(img_dbg, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
      cv2.line(img_dbg, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
      cv2.line(img_dbg, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0) # TODO pokvari se poredak trokuta

      cv2.putText(img_dbg, str(0), pt1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))  

  
      cv2.imshow("asd", img_dbg)
      cv2.waitKey(0)

      triangles_morphed.append(curr)
    
    # TODO region cropping and color interpolation
    
    # i, = np.where( a==value )
    #idx = list(classes).index(var)
    
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

  
