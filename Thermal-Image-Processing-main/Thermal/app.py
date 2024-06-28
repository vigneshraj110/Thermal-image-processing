from flask import Flask, render_template, request
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Load the trained model




# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the HTML form
    file = request.files['image']

    # Save the image to the server
    img_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
    file.save(img_path)
    imgname = file.filename

    # Preprocess the image
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (200, 200))

    x = img.reshape(1, -1)
    x = x / 255
    #Thresholding
    img = cv2.imread(img_path, 0)

# 1-level thresholding
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/thresh1.jpg', thresh1)

    # 2-level thresholding
    ret1, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(thresh2, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('static/thresh2.jpg', thresh2)

    # 3-level thresholding
    ret1, thresh3 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    ret2, thresh3 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/thresh3.jpg', thresh3)

    #CONTOUR 
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image
    contour_img = np.zeros_like(img)


    # Draw the contours on the blank image
    contra = cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    cv2.imwrite('static/contra.jpg', contra)

    #EDGE DETECTION

    img = cv2.imread(img_path, 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('static/edges.jpg', edges)



    #HISTOGRAM EQUALIZER

    img = cv2.imread(img_path, 0)

    equalized_img = cv2.equalizeHist(img)
    cv2.imwrite('static/histogram.jpg', equalized_img)
    img = cv2.imread(img_path, 0)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('static/noise_reduction.jpg', blurred_img)

    img = cv2.imread(img_path, 0)
    # Apply color mapping (Jet colormap)
    colored_img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite('static/colormapped_image.jpg', colored_img)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite('static/erosion.jpg', erosion)

    dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite('static/dilation.jpg', dilation)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bilateral_filtered_img = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imwrite('static/bilateral_filtered.jpg', bilateral_filtered_img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur or other preprocessing if needed
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create a copy of the original image
    img_with_circles = img.copy()

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

    # Draw circles on the copy of the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Save the image with circles
    cv2.imwrite('static/circles_detected.jpg', img_with_circles)


    img = cv2.imread(img_path)

        # Create a mask and rectangle for initialization
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

        # Apply GrabCut
    cv2.grabCut(img, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

        # Modify the mask to create a binary mask for foreground and background
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply the mask to the original image
    result = img * mask2[:, :, np.newaxis]

        # Save the result
    cv2.imwrite('static/grabcut_result.jpg', result)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Harris Corner Detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    img[corners > 0.01 * corners.max()] = [0, 0, 0]  # Highlight corners in red

    

    cv2.imwrite('static/corners.jpg', img)

    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Compute Local Binary Pattern
    radius = 1
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(img, n_points, radius, method="uniform")

    # Convert the LBP image to uint8
    lbp = (lbp * 255).astype(np.uint8)

    cv2.imwrite('static/lbp.jpg', lbp)

    img = cv2.imread(img_path)

    # Apply Laplacian kernel for sharpening
    sharpened_img = cv2.Laplacian(img, cv2.CV_8U)

    # Save the sharpened image
    sharpened_path = 'static/sharpened.jpg'
    cv2.imwrite(sharpened_path, sharpened_img)


    img = cv2.imread(img_path)

    # Apply log transformation
    img_log = np.log1p(img)

    # Convert back to uint8
    img_log = (255 * img_log / np.max(img_log)).astype(np.uint8)

    # Save the processed image
    cv2.imwrite('static/log_transformed.jpg', img_log)



    return render_template('index.html',imgname=imgname)




if __name__ == '__main__':
    app.run(debug=True)