import cv2
import numpy as np
from ex1 import equaliza
from ex2 import realca_caixa_vermelha

# courtesy of: https://pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def test_ex1_equalization():
    img_given = cv2.imread("APS01/img/RinTinTin.jpg", cv2.IMREAD_GRAYSCALE)
    img_submitted = equaliza(img_given)
    print(img_submitted.min(), img_submitted.max())
    assert img_submitted.min() == 0
    assert img_submitted.max() == 250

def test_ex1_compare_img():
    img_given = cv2.imread("APS01/img/RinTinTin.jpg", cv2.IMREAD_GRAYSCALE)
    img_submitted = equaliza(img_given)
    img_expected = cv2.imread("APS01/expected_img/ex1.jpg", cv2.IMREAD_GRAYSCALE)
    err = mse(img_submitted, img_expected)
    print(err)
    assert err < 1

def test_ex2():
    img_given = cv2.imread("APS01/img/cena_canto_sala.jpg")
    img_submitted = realca_caixa_vermelha(img_given)
    img_submitted = img_submitted[:,:,::-1]
    img_expected = cv2.imread("APS01/expected_img/ex2.jpg")
    err = mse(img_submitted, img_expected)
    print(err)
    assert err < 1
