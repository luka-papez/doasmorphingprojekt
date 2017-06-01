from helpers import *
from morphing import *

#print find_facial_landmarks(load_resized_images('images/face1.jpg'))

imgs = load_resized_images("images/face1.jpg", "images/face2.jpg")

#SimpleLinearMorpher().morph(imgs[0], imgs[1])

AdvancedMorphingAlgorithm().morph(imgs[0], imgs[1])
