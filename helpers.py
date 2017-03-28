import cv2
import dlib

"""
  Find the 68 landmark points of a face in an image.
  Parts of code from: http://www.codesofinterest.com/2016/10/getting-dlib-face-landmark-detection.html
  Returns a list of points written as [p.x, p.y].
"""
def get_facial_landmarks(img_src):
  cascade_path = "data/haarcascade_frontalface_default.xml"  
  predictor_path= "data/shape_predictor_68_face_landmarks.dat"  

  # reate the Haar face detector  
  face_detector = cv2.CascadeClassifier(cascade_path)  
 
  # create the landmark predictor  
  landmark_predictor = dlib.shape_predictor(predictor_path)  

  img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
  
  # detect faces in the image  
  faces = face_detector.detectMultiScale(  
      img_gray,  
      scaleFactor=1.05,  
      minNeighbors=5,  
      minSize=(100, 100),  
      flags=0
  )

  x, y, w, h = faces[0]

  # converting the OpenCV rectangle coordinates to Dlib rectangle  
  dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

  # detect landmarks
  detected_landmarks = landmark_predictor(img_src, dlib_rect).parts()  # TODO: mozda radi s crno bijelom
  # convert them to human-manageable form
  landmarks = [[p.x, p.y] for p in detected_landmarks]
  
  """
  # debugging, draw found points
  cv2.rectangle(img_src, (x, y), (x + w, y + h), (0, 255, 0), 2)  
  for (i, point) in enumerate(landmarks):  
    pos = (point[0], point[1])  

    # draw points on the landmark positions
    cv2.circle(img_src, pos, 3, color=(0, 255, 255))  

  cv2.imshow('conan', img_src)
  cv2.waitKey(0)  
  """
    
  return landmarks

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
  Returns images as a list of numpy arrays from given paths.
  The images are resized to be the same size.
"""
def load_images(*paths):
  print "Loading images from: ", paths
  output = []

  # http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?#imread
  for path in paths:
    output.append(cv2.imread(path))

  # resize the images to be the same size
  def fun(acc, curr):
    return min(curr.shape[0], acc[0]), min(curr.shape[1], acc[1])

  new_height, new_width = reduce(fun, output, output[0].shape)
  
  # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize  
  return map(lambda img: cv2.resize(src = img, dsize = (new_height, new_width)), output)
  
  
  
