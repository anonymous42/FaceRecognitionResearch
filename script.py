import numpy
import face_recognition

known_faces = []
input_image = input()

faces_image = face_recognition.load_image_file(input_image)

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