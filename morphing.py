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
  def morph(self, img_src, img_dst, n_steps = 10):
    raise NotImplementedError("Please Implement this method")

