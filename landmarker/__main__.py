import imutils
import numpy 
import cv2
from matplotlib import pyplot
import time
import uuid

figure = pyplot.figure()
face_cascade = cv2.CascadeClassifier('cvmodel/haarcascade_frontalface_default.xml')
img1 = None

def render_image(image):
    global img1
    if img1 == None:
        img1 = pyplot.imshow(image, cmap='gray', vmin=0, vmax=255) 
    else:
        img1.set_data(image)

def write_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('out/%s.jpg' % uuid.uuid4(), image)

def detect(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print "found %d faces" % len(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(
            image,
            (x , y),
            (x + int(w), y + int(h)),
            (255,0,0),
            2
        )
        write_image(image[y:(y + h), x:(x + w)])
        # cv2.imwrite('%s.jpg' % uuid.uuid4(), image[y:(y + h), x:(x + w)])
        
    render_image(image)

# img = cv2.imread('res/multiface1.jpg', cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_h, img_w, _ = img.shape

# stream = cv2.VideoCapture('rtmp://localhost/live/test')
stream = cv2.VideoCapture(0)
pyplot.ion()

while True:
    ok, frame = stream.read()

    if ok:
        print "Retrieve frame!"
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect(frame)
        pyplot.pause(0.1)

# detect(img)

# render_image(img)

pyplot.show()