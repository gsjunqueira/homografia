""" Curso IDASE
    Disciplina de Visão Computacional
    Doscente: Dr. Vinicius Ferreira Vidal,
    Discente: Giovani Santiago Junqueira

    Capture um conjunto de dados com o próprio smartphone, rotacionando o aparelho em torno do
    próprio eixo, como se estivesse capturando uma imagem panorâmica. calcular a homografia e
    fazer a fusão de imagens duas a duas, salvar os resultados, e utilizar os mesmos para fusão
    da imagem final.
"""
import os
import cv2
import homography as homo


def dual_homografia(im_1, im_2, path, passo):
    """ Realiza a homografia entre duas imagens e salva os resultados intermediários e finais.

    Esta função redimensiona as imagens de entrada, detecta e descreve os keypoints,
    corresponde as descrições, calcula a homografia e, finalmente, aplica a homografia
    para realizar a fusão das imagens. Os resultados intermediários e finais são salvos
    em arquivos na pasta especificada.

    Args:
        im_1 (ndarray): A primeira imagem para a homografia.
        im_2 (ndarray): A segunda imagem para a homografia.
        dir (str): O diretório onde os arquivos intermediários e finais serão salvos.
        passo (int): Um valor inteiro que será usado para nomear os arquivos de saída.

    Returns:
        ndarray: A imagem resultante da fusão das duas imagens utilizando homografia.
    """
    h, w = im_1.shape[:2]
    nh = h // 5
    nw = w // 5
    im_1 = cv2.resize(im_1, (nw, nh))
    im_2 = cv2.resize(im_2, (nw, nh))
    keypoint_1, describe_1, im_1_keypoint, im_1_describe = homo.detect_keypoint_describe(im_1)
    keypoint_2, describe_2, im_2_keypoint, im_2_describe = homo.detect_keypoint_describe(im_2)
    correspondence = homo.match_filter_describe(describe_1, describe_2, 0.5)
    result = cv2.drawMatches(im_1, keypoint_1, im_2, keypoint_2, correspondence, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    homografia = homo.calcula_homografia(keypoint_1, keypoint_2, correspondence)
    im_1_homografia = cv2.warpPerspective(im_1, homografia, (im_2.shape[1], im_2.shape[0]))
    result_stitching = homo.stitching_mascara(im_1, im_2, homografia)

    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/kp_{1 + passo}.jpg'), im_1_keypoint)
    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/kp_{2 + passo}.jpg'), im_2_keypoint)
    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/desc_{1 + passo}.jpg'), im_1_describe)
    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/desc_{2 + passo}.jpg'), im_2_describe)
    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/result_{1 + passo / 2}.jpg'), result)
    cv2.imwrite(os.path.join(path, f'datasets/atividade_4/homog_{1 + passo / 2}.jpg'),
                im_1_homografia)

    return result_stitching


def main():
    """ Função principal que executa o pipeline de homografia para um conjunto de imagens.

    Esta função lê várias imagens de um diretório específico, aplica a função
    `dual_homografia` em pares de imagens, e salva os resultados intermediários e finais
    em arquivos na mesma pasta. O objetivo é demonstrar o processo de combinação de múltiplas
    imagens utilizando homografia.

    Returns:
        None
    """
    dir_actual = os.path.dirname(os.path.realpath(__file__))
    print("Diretório do arquivo atual:", dir_actual)
    img_1 = cv2.imread(os.path.join(dir_actual, 'datasets/IMG_3855.jpeg'))
    img_2 = cv2.imread(os.path.join(dir_actual, 'datasets/IMG_3856.jpeg'))
    img_3 = cv2.imread(os.path.join(dir_actual, 'datasets/IMG_3853.jpeg'))
    img_4 = cv2.imread(os.path.join(dir_actual, 'datasets/IMG_3854.jpeg'))

    img_5 = dual_homografia(img_1, img_2, dir_actual, 0)
    cv2.imwrite(os.path.join(dir_actual, 'datasets/Homografia_1.jpg'), img_5)
    img_6 = dual_homografia(img_3, img_4, dir_actual, 2)
    cv2.imwrite(os.path.join(dir_actual, 'datasets/Homografia_2.jpg'), img_6)
    img_7 = dual_homografia(img_5, img_6, dir_actual, 4)
    cv2.imwrite(os.path.join(dir_actual, 'datasets/Homografia_3.jpeg'), img_7)


if __name__ == '__main__':
    main()
