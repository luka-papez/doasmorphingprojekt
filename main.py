import cv2
import helpers
import morphing 
import sys

"""
  Run with: python main.py -s ./images/conan.jpg -d ./images/governor.jpg 
"""
if __name__ == "__main__":
      
  # check for validity of arguments
  import argparse
  parser = argparse.ArgumentParser(prog=sys.argv[0], usage='%(prog)s [options]')
  parser.add_argument('--src', '-s', help='Source image')
  parser.add_argument('--dst', '-d', help='Destination image')
  parser.add_argument('--linear', '-l', help='Use the simple linear algorithm', action='store_true')
  parser.add_argument('--advanced', '-a', help='Use the more advanced algorithm', action='store_true')
  parser.add_argument('--interactive', '-i', help='Use the interactive slider mode', action='store_true')
  parser.add_argument('--steps', '-n', help='Number of morphing steps when saving images')
  
  from sys import argv
  if len(argv) < 2:
    parser.print_help()
    exit(1)

  # parse arguments into a namespace object
  args = parser.parse_args()

  # load the images as numpy arrays
  img_src, img_dst = helpers.load_resized_images(args.src, args.dst)
  
  # declare the morphing algorithm
  morphing_algorithm = None
  if args.advanced:
    morphing_algorithm = morphing.AdvancedMorphingAlgorithm(img_src, img_dst)  
  else:
    morphing_algorithm = morphing.SimpleLinearMorpher(img_src, img_dst)
    
  # define the number of steps
  n_steps = 10
  if args.steps is not None:
    n_steps = int(args.steps)
    
  # in interactive mode there are no steps
  if args.interactive:
    n_steps = 0  
  
  # morph the images
  morphs = morphing_algorithm.morph(n_steps)
  
  # save the results
  if not args.interactive:
   helpers.save_images(morphs)
   print "Saved", n_steps, "images"
  
