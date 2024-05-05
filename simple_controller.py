
# Importacion de librerias
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

# Se definen constantes
MIN_LINE_LENGTH = 1
MAX_LINE_GAP = 7
SPEED = 50

# Funcion para obtener imagen de camara
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Funcion para convertir imagen a escala de grises
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Funcion para desplegar imagen
def display_image(display, image):
    # Convertir imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Se despliega imagen
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Funcion para calcular angulo de giro, basado en resultado de Hough
def calculate_steering_angle(lines, image_width):
    if lines is None or len(lines) == 0: # Si Hough no regresa valores, entonces no hay lineas
        SPEED = 10 # Se reduce la velocidad
        return .21 # Se regresa angulo de giro para que auto se dirija a la derecha

    # Calculo del centro de la linea amarilla
    center_x = np.mean([x1 + x2 for line in lines for x1, _, x2, _ in line]) / len(lines)
    
    # Calculo de la desviacion del centro de la imagen
    deviation = center_x - image_width / 2
    
    # Convertir desviacion a angulo de giro
    steering_angle = deviation / (image_width / 2) * 0.5
    
    return steering_angle

# Funcion main
def main():

    # Se crean las instancias robot y driver
    robot = Car()
    driver = Driver()

    # Se obtiene el timestep del mundo actual
    timestep = int(robot.getBasicTimeStep())

    # Se crea instancia de camara
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Se crea instancia del display
    display_img = Display("display_image")

    while robot.step() != -1:

        # Se obtiene imagen de la camara
        image = get_image(camera)

        # Se convierte imagen de camara a escala de grises
        gray_image = greyscale_cv2(image)

        # Se convierte imagen de camara a RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Se aplica Gaussian Blur
        img_blur = cv2.GaussianBlur(gray_image,(5,5), 0, 0)

        # Se detectan bordes con funcion Canny
        edges = cv2.Canny(img_blur, 150, 200)

        # Se crea la Region Of Interest
        vertices = np.array([[(55,128),(55, 100), (201, 100), (201,128)]], dtype=np.int32)
        img_roi = np.zeros_like(gray_image)
        cv2.fillPoly(img_roi, vertices, 255)
        img_mask = cv2.bitwise_and(edges, img_roi)

        # Deteccion de lineas usando HoughLinesP
        lines = cv2.HoughLinesP(img_mask, 2, np.pi / 180, 22, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

        # Se crea fondo negro del tamaño de la imagen con bordes
        img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)

        # Se dibujan lineas detectadas en la imagen
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 255, 0), 1)  

        # Se calcula el angulo de giro basado en las lineas detectadas
        steering_angle = calculate_steering_angle(lines, img_lines.shape[1])

        # Se muestra imagen original con lineas generadas por algoritmo Hough
        alpha = 1
        beta = 1
        gamma = 1
        img_lane_lines = cv2.addWeighted(img_rgb, alpha, img_lines, beta, gamma)

        # Se despliega imagen procesada con las lineas detectadas
        display_image(display_img, img_lane_lines)

        # Se actualiza angulo de giro
        driver.setSteeringAngle(steering_angle)
       
        # Se actualiza veolicidad del vehiculo
        driver.setCruisingSpeed(SPEED)

if __name__ == "__main__":
    main()
