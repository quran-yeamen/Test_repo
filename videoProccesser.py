import numpy as np
import cv2
import time

# Load classes
with open('..PycharmProjects/videoPross//model/synset_words.txt') as f:
    classes = [line[line.find(' ') + 1:].strip() for line in f]

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe('..PycharmProjects/videoPross//model/modelbvlc_googlenet.prototxt', '..PycharmProjects/videoPross//model/bvlc_googlenet.caffemodel')

# Load video
cap = cv2.VideoCapture('../test/images/Self_Driving.mp4')
if not cap.isOpened():
    print('Cannot open video stream')
    exit()

# Output file for processed video
output = cv2.VideoWriter(
    '../output/self_driving_output.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    20.0,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 5 frames for efficiency
    if frame_count % 5 == 0:
        # Preprocess frame for DNN model
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123))
        net.setInput(blob)

        # Measure inference time
        start_time = time.time()
        outp = net.forward()
        end_time = time.time()
        print(f"Inference Time per Frame: {end_time - start_time:.2f} seconds")

        # Get top prediction
        top_idx = np.argmax(outp[0])
        top_class = classes[top_idx]
        top_prob = outp[0][top_idx] * 100
        txt = f'Top Prediction: {top_class} ({top_prob:.2f}%)'

        # Overlay text on frame
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame with prediction
    cv2.imshow('Processed Video', frame)

    # Write frame to output file
    output.write(frame)

    # Press 'ESC' to break the loop
    if cv2.waitKey(25) & 0xFF == 27:
        break

    frame_count += 1

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
