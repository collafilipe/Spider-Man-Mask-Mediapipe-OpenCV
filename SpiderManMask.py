#1
# Importing the required libraries: cv2, mediapipe, and numpy
import cv2
import mediapipe as mp
import numpy as np

#2
# Initializing video capture (default camera)
frm = cv2.VideoCapture(0)

#3
# Initializing FaceMesh
mp_face_mesh = mp.solutions.face_mesh

#4
# Set of landmarks for the nose and eyes
FACEMESH_NOSE = frozenset([168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64])
LANDMARKS_REF = (33, 263)  # Landmarks for the corners of the eyes (right and left)

#5
# Function to calculate the distance between two points
def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

#6
# Function to calculate the angle between two points
def calcular_angulo(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return np.degrees(np.arctan2(delta_y, delta_x))

# Function to apply the Spider-Man mask with rotation
def aplicar_mascara(img, img_mask, nose_landmarks, escala, angulo):
    #7
    # Calculating the center of the nose
    nose_x = int(np.mean([point[0] for point in nose_landmarks]))
    nose_y = int(np.mean([point[1] for point in nose_landmarks]))

    #8
    # Resizing the mask based on scale
    h, w, _ = img.shape
    mascara_resized = cv2.resize(img_mask, (int(w * escala), int(h * escala)))

    #9
    # Rotating the mask
    centro = (mascara_resized.shape[1] // 2, mascara_resized.shape[0] // 2)
    matriz_rotacao = cv2.getRotationMatrix2D(centro, -angulo, 1)
    mascara_rotated = cv2.warpAffine(mascara_resized, matriz_rotacao, (mascara_resized.shape[1], mascara_resized.shape[0]), flags=cv2.INTER_LINEAR)

    #10
    # Splitting the alpha channel (transparency)
    b, g, r, a = cv2.split(mascara_rotated)
    mascara_rgb = cv2.merge((b, g, r))

    #11
    # Initial position of the mask
    y_offset = nose_y - mascara_resized.shape[0] // 2
    x_offset = nose_x - mascara_resized.shape[1] // 2

    #12
    # Ensuring the mask stays within image boundaries
    y1, y2 = max(0, y_offset), min(h, y_offset + mascara_resized.shape[0])
    x1, x2 = max(0, x_offset), min(w, x_offset + mascara_resized.shape[1])

    #13
    # Adjusting the mask and alpha channel for the overlapping region
    mask_region = mascara_rgb[: y2 - y1, : x2 - x1]
    alpha_region = a[: y2 - y1, : x2 - x1] / 255.0

    #14
    # Overlaying the mask using the alpha channel
    for c in range(3):  # For each color channel (B, G, R)
        img[y1:y2, x1:x2, c] = (
            alpha_region * mask_region[:, :, c] +
            (1 - alpha_region) * img[y1:y2, x1:x2, c]
        )

#15
# Loading the mask image with an alpha channel
img_mask = cv2.imread("./Assets/pngegg.png", cv2.IMREAD_UNCHANGED)

#16
# Configuring the FaceMesh detector
with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    
    #17
    # Running the loop for video capture
    while True:
        success, img = frm.read()

        #18
        # Converting the image to RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #19
        # Processing facial landmarks
        results = face_mesh.process(image_rgb)

        #20
        # Converting back to BGR
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        #21
        # Checking if facial landmarks are detected
        if results.multi_face_landmarks:

            #22
            # Converting the image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            #23
            # Looping through each face
            for face_landmarks in results.multi_face_landmarks:

                #24
                # Getting face coordinates
                h, w, _ = img.shape

                #25
                # Getting all the landmarks
                landmarks = [
                    (
                        #26
                        # Converting normalized coordinates to pixel coordinates
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h),
                    )

                    #27
                    # Looping through all the landmarks
                    for i in range(468)
                ]

                #28
                # Coordinates for reference distance and angle
                ref_distancia = calcular_distancia(

                    #29
                    # Getting the landmarks for the corners of the eyes (right and left)
                    landmarks[LANDMARKS_REF[0]], landmarks[LANDMARKS_REF[1]]
                )

                #30
                # Calculating the angle based on the reference landmarks (eyes)
                angulo = calcular_angulo(landmarks[LANDMARKS_REF[0]], landmarks[LANDMARKS_REF[1]])

                #31
                # Adjusting scale based on distance between the eyes
                escala = (ref_distancia / w) * 3

                #32
                # Getting nose landmarks for mask placement, comparing with the nose landmarks
                nose_landmarks = [landmarks[i] for i in FACEMESH_NOSE]

                #33
                # Applying the mask with rotation and scaling based on the nose landmarks
                aplicar_mascara(img, img_mask, nose_landmarks, escala, angulo)

        #34
        # Displaying the image with the mask
        cv2.imshow("Spider-Man Mask", img)

        #35
        # Breaking the loop and closing the window when 'q' is pressed
        # Closing when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

#36
# Releasing the video capture and closing the window when the loop ends
frm.release()
cv2.destroyAllWindows()
