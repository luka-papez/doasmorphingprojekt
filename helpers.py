import cv2

"""
  Performs linear interpolation between two images, returning 
  @n_steps intermediate images as a list
"""

def linear_interpolation(img_src, img_dst, n_steps = 10):
  output = []
  
  # repeat n_steps times
  for i in xrange(0, n_steps):
    # http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#addweighted
    curr = cv2.addWeighted(img_src, 1 - float(i) / float(n_steps - 1), img_dst,  float(i) / float(n_steps - 1), 0)
    output.append(curr)
    
  return output
  
"""
  Returns two images as numpy arrays from given paths.
  The images are resized to be the same size
"""
def load_images(path_src, path_dst):
  print "Loading images from: ", path_src, path_dst

  # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?#imread
  img_src = cv2.imread(path_src)
  img_dst = cv2.imread(path_dst)

  # resize the images to be the same size
  new_height, new_width = min(img_src.shape[0], img_dst.shape[0]), min(img_src.shape[1], img_dst.shape[1])
  
  # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
  img_src = cv2.resize(src = img_src, dsize = (new_height, new_width))
  img_dst = cv2.resize(src = img_dst, dsize = (new_height, new_width))
  
  return img_src, img_dst
