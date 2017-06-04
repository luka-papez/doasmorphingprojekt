from helpers import *
from morphing import *

#print find_facial_landmarks(load_resized_images('images/face1.jpg'))

imgs = load_resized_images("images/face2.jpg", "images/face1.jpg")

#SimpleLinearMorpher().morph(imgs[0], imgs[1])

out = AdvancedMorphingAlgorithm(imgs[0], imgs[1]).morph(10)
helpers.save_images(out)
