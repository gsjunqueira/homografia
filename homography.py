""" Curso IDASE
    Disciplina de Visão Computacional
    Doscente: Dr. Vinicius Ferreira Vidal,
    Discente: Giovani Santiago Junqueira

    Capture um conjunto de dados com o próprio smartphone, rotacionando o aparelho em torno do
    próprio eixo, como se estivesse capturando uma imagem panorâmica. calcular a homografia e
    fazer a fusão de imagens duas a duas, salvar os resultados, e utilizar os mesmos para fusão
    da imagem final.
"""
import cv2
import numpy as np


def detect_keypoint_describe(image):
    """ Detecta pontos-chave (keypoints) e calcula os descritores de uma imagem usando o algoritmo
    SIFT.

    Args:
        image (numpy.ndarray): Imagem de entrada em escala de cinza.

    Returns:
        tuple: Contém os seguintes elementos:
            - keypoint (list): Lista de objetos KeyPoint, representando os pontos-chave detectados.
            - describe (numpy.ndarray): Array de descritores correspondentes aos pontos-chave.
            - image_keypoint (numpy.ndarray): Imagem com os pontos-chave desenhados.
            - image_describe (numpy.ndarray): Imagem com círculos coloridos indicando a área de cada
            ponto-chave.
    """
    sift = cv2.SIFT_create()
    keypoint, describe = sift.detectAndCompute(image, None)
    image_keypoint = cv2.drawKeypoints(image, keypoint, None)
    image_describe = image.copy()
    for kp in keypoint:
        x, y = kp.pt
        size = kp.size
        color = tuple([int(c) for c in np.random.randint(0, 255, 3)])
        cv2.circle(image_describe, (int(x), int(y)), int(size), color, 2)

    return keypoint, describe, image_keypoint, image_describe


def match_filter_describe(describe_1, describe_2, threshold=0.9):
    """ Filtra as correspondências (matches) entre dois conjuntos de descritores usando força bruta
    e o teste de razão.

    Args:
        describe_1 (numpy.ndarray): Array de descritores da primeira imagem.
        describe_2 (numpy.ndarray): Array de descritores da segunda imagem.
        threshold (float, opcional): Fator de limiar para o teste de razão de Lowe. O valor padrão é
        0.9.

    Returns:
        list: Lista de objetos DMatch que passaram pelo filtro de teste de razão.
    """
    bf = cv2.BFMatcher()
    correspondence = bf.knnMatch(describe_1, describe_2, k=2)
    boas_correspondence = []
    for m, n in correspondence:
        if m.distance < threshold * n.distance:
            boas_correspondence.append(m)

    return boas_correspondence


def calcula_homografia(k_pts_1, k_pts_2, corresp):
    """ Calcula a matriz de homografia entre dois conjuntos de pontos-chave correspondentes.

    Args:
        k_pts_1 (list): Lista de pontos-chave da primeira imagem.
        k_pts_2 (list): Lista de pontos-chave da segunda imagem.
        corresp (list): Lista de correspondências (objetos DMatch) entre os pontos-chave.

    Returns:
        numpy.ndarray: Matriz de homografia (3x3) que mapeia os pontos da primeira imagem para a
        segunda.
    """
    pontos1 = np.float32([k_pts_1[m.queryIdx].pt for m in corresp]).reshape(-1, 1, 2)
    pontos2 = np.float32([k_pts_2[m.trainIdx].pt for m in corresp]).reshape(-1, 1, 2)
    homografia, _ = cv2.findHomography(pontos1, pontos2, cv2.RANSAC)

    return homografia


def stitching_mascara(im1, im2, h):
    """ Realiza o mapeamento e a junção de duas imagens usando a matriz de homografia.

    Args:
        im1 (numpy.ndarray): Primeira imagem (base) a ser mapeada e combinada.
        im2 (numpy.ndarray): Segunda imagem a ser combinada.
        h (numpy.ndarray): Matriz de homografia que mapeia a primeira imagem para a segunda.

    Returns:
        numpy.ndarray: Imagem resultante da combinação das duas imagens.
    """
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners1_homografia = cv2.perspectiveTransform(corners1, h)
    corners = np.concatenate((corners1_homografia, corners2), axis=0)
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
    t = [-x_min, -y_min]
    h_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    im1_homografia = cv2.warpPerspective(im1, h_t @ h, (x_max - x_min, y_max - y_min))
    im1_homografia[t[1]:h2 + t[1], t[0]:w2 + t[0]] = im2

    return im1_homografia.astype(np.uint8)
