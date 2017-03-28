import cv2
import helpers

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
    return helpers.linear_interpolation(img_src, img_dst, n_steps)

"""
  TODO: write better program description
  Run with: python morphing.py -s ./images/conan.jpeg -d ./images/governor.jpg 
"""
if __name__ == "__main__":
      
  # check for validity of arguments
  import argparse
  parser = argparse.ArgumentParser(prog='IMAGE MORPHING', usage='%(prog)s [options]')
  parser.add_argument('--src', '-s', help='Source image')
  parser.add_argument('--dst', '-d', help='Destination image')
  
  from sys import argv
  if len(argv) < 2:
    parser.print_help()
    exit(1)

  # parse arguments into a namespace object
  args = parser.parse_args()

  # load the images as numpy arrays
  img_src, img_dst = helpers.load_images(args.src, args.dst)
  
  # declare the morphing algorithm
  morphing_algorithm = SimpleLinearMorpher()
  
  # morph the images
  morphs = morphing_algorithm.morph(img_src, img_dst, n_steps = 10)
  
  # save the morphs to file as jpg files
  for (i, morph) in enumerate(morphs):
    # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?#imwrite
    cv2.imwrite("morphs/morph" + str(i) + ".jpg", morph)
  
  # indefinetly wait for keypress
  cv2.waitKey(0)
  
