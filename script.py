import cv2
import numpy
import face_recognition

video_capture = cv2.VideoCapture(0)
known_faces = []

while True:
	_, frame_image = video_capture.read()
	frame_image = frame_image[:, :, ::-1]
	faces_image = face_recognition.load_image_file(frame_image)
	bbox = face_recognition.face_locations(faces_image)
	desc = face_recognition.face_encodings(faces_image, known_face_locations=bbox)
	for i in desc:
		dist = face_recognition.face_distance(known_faces, i)
		if known_faces == []:
			known_faces.append(i)
			print("Welcome, User#"+str(len(known_faces)-1))
			continue
		kk = numpy.argmin(dist)
		if dist[kk] < 0.6:
			print("Hello again, User#"+str(kk))
		else:
			known_faces.append(i)
			print("Welcome, User#"+str(len(known_faces)-1))
video_capture.release()