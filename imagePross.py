import numpy as np
import cv2
img = cv2.imread('../test/images/sign22.jpg')

#img = cv2.imread('../test/images/car.jpg')

all_rows = open('../test/model/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv2.dnn.readNetFromCaffe('../test/model/bvlc_googlenet.prototxt', '../test/model/bvlc_googlenet.caffemodel')

blob = cv2.dnn.blobFromImage(img, 1, (224,224))

net.setInput(blob)

outp = net.forward()
#  top five probability.
idx = np.argsort(outp[0])[::-1][:5]

#  print the top five probability.
for(i,id) in enumerate(idx):
    print('{}. {} ({}): probability {:.3}%'.format(i+1, classes[id], id, outp[0][id]*100))
#print (outp)
#for(i,c) in enumerate(classes):
#    if i==4:
#        break
#    print(i,c)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
