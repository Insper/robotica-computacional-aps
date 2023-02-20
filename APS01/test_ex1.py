import cv2
from ex1 import equaliza

def test_ex1():
    img_given = cv2.imread("APS01/img/RinTinTin.jpg", cv2.IMREAD_GRAYSCALE)
    img_submitted = equaliza(img_given)
    print(img_submitted.min(), img_submitted.max())
    assert img_submitted.min() == 0
    assert img_submitted.max() == 250
