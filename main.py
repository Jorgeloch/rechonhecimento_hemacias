import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, filters, exposure
from skimage.color import rgb2hsv
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

def process_malarian_cells (malarian_bin):
    # processing the binarized image to remove noise, fill holes and improve the cells recognition
    opened_image = ((morphology.binary_opening(malarian_bin, morphology.disk(5)).astype(np.uint8)) * 255).astype("uint8")
    closed_image = ((morphology.binary_closing(opened_image, morphology.disk(20)).astype(np.uint8)) * 255).astype("uint8")
    final_image = ((morphology.binary_opening(closed_image, morphology.disk(20)).astype(np.uint8)) * 255).astype("uint8")
    # removing the edges
    edges = (filters.sobel(final_image)*255).astype("uint8")

    return (final_image, edges)

def process_rb_cells (malarian_cells, rbc_bin):
    # removing malarian cells from the hemogram binarized image
    mask = (morphology.binary_dilation(malarian_cells, morphology.disk(20)) * 255).astype("uint8")
    malarian_removed = rbc_bin - mask

    # processing the resultant image
    final_image = ((morphology.binary_opening(malarian_removed, morphology.disk(20)).astype(np.uint8)) * 255).astype("uint8")
    edges = (filters.sobel(final_image)*255).astype("uint8")

    return(final_image, edges)

def detect_malarian_cells (malarian_cells_edges, malarian_filled):
    # Detect two radii
    hough_radii = np.arange(30, 80, 2)
    hough_res = hough_circle(malarian_cells_edges, hough_radii)

    # Select the possible circles detected
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 50, 50, total_num_peaks=60)

    # Draw them
    color_image = original_img
    circles_drawned = 0
    for center_y, center_x, radius in zip(cy, cx, radii):

        # checking if the possible center is inside a cell usign the binarized image filled
        if (malarian_filled[center_y,center_x] != 0):

            # drawing the circle above the cells
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=color_image.shape)
            color_image[circy, circx] = (220, 20, 20)
            circy, circx = circle_perimeter(center_y, center_x, radius+1, shape=color_image.shape)
            color_image[circy, circx] = (220, 20, 20)
            circy, circx = circle_perimeter(center_y, center_x, radius-1, shape=color_image.shape)
            color_image[circy, circx] = (220, 20, 20)

            # counting the drawned circles (i.e. the number of the cells detected)
            circles_drawned += 1

    if (circles_drawned == 0):
        print("it wasn't found malarian cells in this image!")
    else:
        print(f"it was found {circles_drawned} malarian cells in this image")

    return color_image

def detect_rb_cells (rb_cells_edges, rb_cells_filled, malarian_detected):
    # Detect two radii
    hough_radii = np.arange(30, 70, 2)
    hough_res = hough_circle(rb_cells_edges, hough_radii)

    # Select possible circles detected
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, 50, 50, total_num_peaks=200)

    # Draw them
    color_image = malarian_detected

    circles_drawned = 0
    
    for center_y, center_x, radius in zip(cy, cx, radii):

        # checking if the possible center is inside a cell usign the binarized image filled
        if (rb_cells_filled[center_y,center_x] != 0):

            # drawing the circle above the cells
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=color_image.shape)
            color_image[circy, circx] = (20, 220, 20)
            circy, circx = circle_perimeter(center_y, center_x, radius + 1, shape=color_image.shape)
            color_image[circy, circx] = (20, 220, 20)
            circy, circx = circle_perimeter(center_y, center_x, radius - 1, shape=color_image.shape)
            color_image[circy, circx] = (20, 220, 20)

            # counting the drawned circles (i.e. the number of the cells detected)
            circles_drawned += 1

    if (circles_drawned == 0):
        print("it wasn't found red blood cells in this image!\n")
    else:
        print(f"it was found {circles_drawned} red blood cells in this image")
    
    return color_image

# listing the images that will be used on this test
images = ["0ab56f9a-846d-49e2-a617-c6cc477fdfad", "0ac747cd-ff32-49bf-bc1a-3e9b7702ce9c", "0b04ec46-5119-4cda-8c35-c4e5b6f0eed0",
           "0b923ab7-ebff-4079-a4bb-af7da89f374e", "00c8364b-8c85-4502-bcfe-64736fe76815", "0ca25c88-457f-4f03-bbc1-98fb6663f1d1",
           "0ceb4539-5c4c-487d-9826-452a88b5d537", "0ceb4539-5c4c-487d-9826-452a88b5d537", "0ceb4539-5c4c-487d-9826-452a88b5d537",
           "00d04a90-80e5-4bce-9511-1b64eabb7a47", "0d7bf56f-3b5a-40bd-971c-2ca33dd89b2c", "0d095f3a-9243-472b-90b4-0ce8309e778c"]

for i in range(0,12):
    path = f"archive/images/{images[i]}.png"
    # getting the image
    original_img = io.imread(path) 
    
    _,axis = plt.subplots(1,2)
    # plotting the original image
    axis[0].set_title("ORIGINAL")
    axis[0].axis("off")
    axis[0].imshow(original_img)

    # turning into a hsv image
    img_hsv = rgb2hsv(original_img)

    # splitting the chanels of the hsv image 
    img_hue = (img_hsv[:,:,0]*255).astype("uint8")
    img_saturation = (img_hsv[:,:,1]*255).astype("uint8")
    img_value = (img_hsv[:,:,2]*255).astype("uint8")

    # equalizing the H channel of the hsv image
    equalized_image = (exposure.equalize_hist(img_hue) * 255).astype("uint8")

    # binarizing the equalized image of the hue channel
    binarized_malaria = equalized_image.copy()

    binarized_malaria[binarized_malaria < 245] = 0
    binarized_malaria[binarized_malaria > 0] = 255

    # binarizing the original image of the hue channel
    binarized_hue_image = img_hue.copy()

    binarized_hue_image[binarized_hue_image < 120] = 0
    binarized_hue_image[binarized_hue_image > 0] = 255

    # processing the binarized image of the malarian cells
    final_malarian_cells, final_malarian_cells_edges = process_malarian_cells(binarized_malaria)

    # removing the malarian cells and processing the image of the red blood cells
    final_blood_cells, final_blood_cells_edges = process_rb_cells(final_malarian_cells, binarized_hue_image)

    # applying the Circular Hough Transform to the malarian image
    malarian_detected = detect_malarian_cells(final_malarian_cells_edges, final_malarian_cells)

    # applying the Circular Hough Transform to the malarian image
    rb_cells_detected = detect_rb_cells(final_blood_cells_edges, final_blood_cells, malarian_detected)

    # plotting the image with the detections
    axis[1].set_title("DETECTION")
    axis[1].axis("off")
    axis[1].imshow(rb_cells_detected)

plt.show()