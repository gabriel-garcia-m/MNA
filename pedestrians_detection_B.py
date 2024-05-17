import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import io
from skimage.transform import resize

# Funcion para extraer características del HOG
def extract_features(image):
    fd, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2),   transform_sqrt = False,
                                               visualize = True,
                                               feature_vector = True)
    return fd

# Se cargan ejemplos positivos (personas) del dataset
def load_positive_samples(directory):
    positive_samples = []
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = io.imread(os.path.join(directory, filename))
            img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #resize image
            resized_img = resize(img_gray, (128,64))
            positive_samples.append(resized_img)
    return positive_samples

# Se cargan ejemplos negativos (no personas) del dataset
def load_negative_samples(directory):
    negative_samples = []
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image = io.imread(os.path.join(directory, filename))
            img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #resize image
            resized_img = resize(img_gray, (128,64))
            negative_samples.append(resized_img)
    return negative_samples

# Se carga el dataset
def load_dataset(pos_dir, neg_dir):
    X_pos = np.array([extract_features(img) for img in load_positive_samples(pos_dir)])
    y_pos = np.ones(X_pos.shape[0])
    X_neg = np.array([extract_features(img) for img in load_negative_samples(neg_dir)])
    y_neg = np.zeros(X_neg.shape[0])

    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    return X, y

# Funcion para entrenar el modelo
def train_svm(X_train, y_train):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    return svm

# Funcion para probar el modelo
def test_svm(svm, X_test, y_test):
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión:", accuracy)

# Funcion para detectar peatones y dibujar rectangulo
def detect_pedestrians(image, svm_model):
    _, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),   transform_sqrt = False,
                                               visualize = True,
                                               feature_vector = True)
    hog_image = hog_image.astype("uint8")

    # Detectar peatones
    detected_pedestrians = []
    for y in range(0, hog_image.shape[0], 128):
        for x in range(0, hog_image.shape[1], 64):
            roi = hog_image[y:y+128, x:x+64]
            fd = extract_features(roi)
            pred = svm_model.predict([fd])
            if pred == 1:
                detected_pedestrians.append((x, y, x + 64, y + 128))

    # Se dibuja rectangulo sobre la imagen procesada
    for (x1, y1, x2, y2) in detected_pedestrians:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

# Rutas del dataset de personas, en este caso se utiliza el de INRIA
pos_dir = 'INRIAPerson/Train/pos'
neg_dir = 'INRIAPerson/Train/neg'

# Carga del dataset
X, y = load_dataset(pos_dir, neg_dir)

# Se divido dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se entrena el modelo
svm_model = train_svm(X_train, y_train)
print('SVM Model features', svm_model)

# Se prueba el modelo
test_svm(svm_model, X_test, y_test)

# Se carga una imagen de prueba
test_image = cv2.imread('test_final.png')
test_image_resized = resize(test_image, (128,64))
test_image_resized = test_image_resized.astype("uint8")

# Conversion a escala de grises de la imagen de prueba
gray_image = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)

# Ejecucion de funcion para detectar peatones
result_image = detect_pedestrians(gray_image, svm_model)

# Se muestra el resultado
cv2.imshow('Pedestrian Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
