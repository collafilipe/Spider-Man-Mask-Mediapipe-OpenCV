#1
import cv2
import mediapipe as mp
import numpy as np

#2
# Inicializando a captura de vídeo (câmera padrão)
frm = cv2.VideoCapture(0)

#3
# Inicializando o FaceMesh
mp_face_mesh = mp.solutions.face_mesh

#4
# Conjunto de landmarks para o nariz e olhos
FACEMESH_NOSE = frozenset([168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64])
LANDMARKS_REF = (33, 263)  # Landmarks dos cantos dos olhos (direito e esquerdo)

#5
# Função para calcular a distância entre dois pontos
def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

#6
# Função para calcular o ângulo entre dois pontos
def calcular_angulo(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return np.degrees(np.arctan2(delta_y, delta_x))


# Função para aplicar a máscara do Homem-Aranha com rotação
def aplicar_mascara(img, img_mask, nose_landmarks, escala, angulo):
    # Calculando o centro do nariz

    #7
    nose_x = int(np.mean([point[0] for point in nose_landmarks]))
    nose_y = int(np.mean([point[1] for point in nose_landmarks]))

    #8
    # Redimensionando a máscara com base na escala
    h, w, _ = img.shape
    mascara_resized = cv2.resize(img_mask, (int(w * escala), int(h * escala)))

    #9
    # Rotacionando a máscara
    centro = (mascara_resized.shape[1] // 2, mascara_resized.shape[0] // 2)
    matriz_rotacao = cv2.getRotationMatrix2D(centro, -angulo, 1)
    mascara_rotated = cv2.warpAffine(mascara_resized, matriz_rotacao, (mascara_resized.shape[1], mascara_resized.shape[0]), flags=cv2.INTER_LINEAR)

    #10
    # Separando o canal alfa (transparência)
    b, g, r, a = cv2.split(mascara_rotated)
    mascara_rgb = cv2.merge((b, g, r))

    #11
    # Posição inicial da máscara
    y_offset = nose_y - mascara_resized.shape[0] // 2
    x_offset = nose_x - mascara_resized.shape[1] // 2

    #12
    # Garantindo que a máscara não ultrapasse as bordas
    y1, y2 = max(0, y_offset), min(h, y_offset + mascara_resized.shape[0])
    x1, x2 = max(0, x_offset), min(w, x_offset + mascara_resized.shape[1])

    #13
    # Ajustando a máscara e o canal alfa para a região de sobreposição
    mask_region = mascara_rgb[: y2 - y1, : x2 - x1]
    alpha_region = a[: y2 - y1, : x2 - x1] / 255.0

    #14
    # Sobrepondo a máscara usando o canal alfa
    for c in range(3):  # Para cada canal de cor (B, G, R)
        img[y1:y2, x1:x2, c] = (
            alpha_region * mask_region[:, :, c] +
            (1 - alpha_region) * img[y1:y2, x1:x2, c]
        )

#15
# Carregando a imagem da máscara com canal alfa
img_mask = cv2.imread("./Assets/pngegg.png", cv2.IMREAD_UNCHANGED)

# Configurando o detector FaceMesh
with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    
    while True:
        success, img = frm.read()

        # Convertendo para RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processando os landmarks faciais
        results = face_mesh.process(image_rgb)

        # Convertendo de volta para BGR
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            for face_landmarks in results.multi_face_landmarks:
                # Pegando as coordenadas do rosto
                h, w, _ = img.shape
                landmarks = [
                    (
                        int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h),
                    )
                    for i in range(468)
                ]

                # Coordenadas para a distância e ângulo de referência
                ref_distancia = calcular_distancia(
                    landmarks[LANDMARKS_REF[0]], landmarks[LANDMARKS_REF[1]]
                )
                angulo = calcular_angulo(landmarks[LANDMARKS_REF[0]], landmarks[LANDMARKS_REF[1]])

                # Ajustar escala com base na distância
                escala = (ref_distancia / w) * 3

                # Pegando os landmarks do nariz
                nose_landmarks = [landmarks[i] for i in FACEMESH_NOSE]

                # Aplicando a máscara com rotação
                aplicar_mascara(img, img_mask, nose_landmarks, escala, angulo)

        # Exibindo a imagem
        cv2.imshow("Spider-Man Mask", img)

        # Encerrando ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Liberando a captura de vídeo e fechando a janela
frm.release()
cv2.destroyAllWindows()
