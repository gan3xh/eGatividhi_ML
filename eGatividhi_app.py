import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as pt
from skimage.metrics import structural_similarity as ssim

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def stitchImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) 
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) 
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    stitched_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min)) 
    img2only = stitched_img.copy()
    stitched_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    img2_binary = np.ones(img2.shape[:2])
    img2_binary = cv2.warpPerspective(img2_binary, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    img1only = np.zeros((y_max-y_min, x_max-x_min, 3), dtype=np.uint8)
    img1only[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    img1_binary = np.zeros((y_max-y_min, x_max-x_min))
    img1_binary[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = 1
    return stitched_img, img1_binary, img1only, img2_binary, img2only

def imageAlignment(img1, img2, surfHessianThreshold, goodMatchPercent):
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    surf = cv2.xfeatures2d.SURF_create(surfHessianThreshold)
    reference_keypoints, reference_descriptor = surf.detectAndCompute(img1Gray, None)
    toAlign_keypoints, toAlign_descriptor = surf.detectAndCompute(img2Gray, None)
    referenceKP = cv2.drawKeypoints(img1.copy(), reference_keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    toAlignKP = cv2.drawKeypoints(img2.copy(), toAlign_keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    print("\nReference image keypoints detected: ", len(reference_keypoints))
    print("Target image keypoints detected: ", len(toAlign_keypoints))
    
    bf = cv2.BFMatcher(crossCheck = True)
    matches = bf.match(toAlign_descriptor, reference_descriptor)
    matches = sorted(matches, key = lambda x : x.distance)
    numGoodMatches = int(len(matches) * goodMatchPercent)
    matches = matches[: numGoodMatches]
    
    print("\nMatching keypoints found: ", len(matches))
    
    result = cv2.drawMatches(img2.copy(), toAlign_keypoints, img1.copy(), reference_keypoints, matches, None, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
    
    reference_points = np.zeros((len(matches), 2), dtype = np.float32)
    toAlign_points = np.zeros((len(matches), 2), dtype = np.float32)
    
    for (i, match) in enumerate(matches):
        reference_points[i] = reference_keypoints[match.trainIdx].pt
        toAlign_points[i] = toAlign_keypoints[match.queryIdx].pt
    
    homography, _ = cv2.findHomography(toAlign_points, reference_points, cv2.RANSAC)
    height, width, _ = img1.shape
    alignedImg = cv2.warpPerspective(img2, homography, (width, height))
    
    stitched_img, img1_binary, img1only, alignedImg_binary, alignedImgOnly = stitchImages(img1, img2, homography)
    
    [rows, cols] = img1_binary.shape
    overlapped = np.zeros((rows, cols), dtype = np.uint8)
    for i in range(rows):
        for j in range(cols):
            if img1_binary[i, j] == 1 and alignedImg_binary[i, j] == 1:
                overlapped[i, j] = 255
    
    img1only_overlapped = cv2.bitwise_and(img1only, img1only, mask=overlapped)
    alignedImg_overlapped = cv2.bitwise_and(alignedImgOnly, alignedImgOnly, mask=overlapped)
    
    return img1only_overlapped, alignedImg_overlapped

def SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize):
    img1HSV = cv2.cvtColor(img1only_overlapped, cv2.COLOR_RGB2HSV)
    _, _, img1Gray = cv2.split(img1HSV)
    alignedImgHSV = cv2.cvtColor(alignedImg_overlapped, cv2.COLOR_RGB2HSV)
    _, _, alignedImgGray = cv2.split(alignedImgHSV)
    img1Blur = cv2.blur(img1Gray, winSize)
    alignedImgBlur = cv2.blur(alignedImgGray, winSize)
    (ssim_score, SSIMimg) = ssim(img1Blur, alignedImgBlur, full=True)
    SSIMimg = (SSIMimg * 255).astype("uint8")
    return ssim_score, SSIMimg

def processDifferences(img1only_overlapped, alignedImg_overlapped, imageWidth, SSIMimg, resizeFactor, winSize):
    MIN_CONTOUR_AREA = (5.5 * imageWidth - 10000) // resizeFactor
    
    SSIMimg = cv2.medianBlur(SSIMimg, winSize[0])
    _, diffBinary = cv2.threshold(SSIMimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    diffContours, _ = cv2.findContours(diffBinary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    mask = np.zeros(alignedImg_overlapped.shape, dtype='uint8')
    detectedChanges = alignedImg_overlapped.copy()
    
    for c in diffContours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 2)
            cv2.drawContours(detectedChanges, [c], 0, (255, 0, 0), 2)
    
    return detectedChanges

def main():
    print("\n" + "="*50)
    print(" Construction Progress Detection System")
    print("="*50)
    
    print("\n--- Main Menu ---\n")
    print("1) Image Alignment")
    print("2) Alignment with Change Detection")
    print("\n0) Exit")
    
    while True:
        try:
            userInput = int(input("\nSelect an option: "))
            if 0 <= userInput < 3:
                if userInput == 0:
                    print("\nExiting program. Goodbye!")
                    sys.exit()
                else:
                    resizeInput = 0
                    break
            else:
                print("\nError: Please select a valid option")
        except ValueError:
            print("\nError: Please enter a number")
    
    if userInput == 2:
        print("\n--- Resize Options ---\n")
        print("1) Original Size")
        print("2) Resize Input Images")
        print("3) Resize SSIM Output")
        print("\n0) Exit")
        
        while True:
            try:
                resizeInput = int(input("\nSelect resize option: "))
                if 0 <= resizeInput < 4:
                    if resizeInput == 0:
                        print("\nExiting program. Goodbye!")
                        sys.exit()
                    else:
                        resizeFactor = 1
                        break
                else:
                    print("\nError: Please select a valid option")
            except ValueError:
                print("\nError: Please enter a number")
        
        if resizeInput != 1:
            print("\n--- Size Options ---\n")
            print("1) 25% of original")
            print("2) 12.5% of original")
            print("3) 10% of original")
            print("\n0) Exit")
            
            while True:
                try:
                    sizeInput = int(input("\nSelect size option: "))
                    if 0 <= sizeInput < 4:
                        if sizeInput == 1:
                            resizeFactor = 4
                            break
                        elif sizeInput == 2:
                            resizeFactor = 8
                            break
                        elif sizeInput == 3:
                            resizeFactor = 10
                            break
                        else:
                            print("\nExiting program. Goodbye!")
                            sys.exit()
                    else:
                        print("\nError: Please select a valid option")
                except ValueError:
                    print("\nError: Please enter a number")
    
    print("\n--- Image Sets ---\n")
    print("1)  SET_B (24/07/2020 & 27/08/2020)")
    print("2)  SET_B (02/09/2020 & 07/10/2020)")
    print("3)  SET_B (25/07/2020 & 14/08/2020)")
    print("4)  SET_B (17/08/2020 & 07/09/2020)")
    print("5)  SET_B (09/09/2020 & 01/10/2020)")
    print("6)  SET_A - West (30/06/2020 & 10/07/2020)")
    print("7)  SET_A - East (17/07/2020 & 24/07/2020)")
    print("8)  SET_A - Overall (21/08/2020 & 28/08/2020)")
    print("9)  SET_A - Overall (04/09/2020 & 11/09/2020)")
    print("10) SET_A - Overall Front (25/09/2020 & 10/10/2020)")
    print("\n0) Exit")
    
    while True:
        try:
            imageInput = int(input("\nSelect image set: "))
            if imageInput >= 0 and imageInput < 11:
                if imageInput == 1:
                    filename1 = "SET_B_1_1.jpg"
                    filename2 = "SET_B_1_2.jpg"
                    break
                elif imageInput == 2:
                    filename1 = "SET_B_2_1.jpg"
                    filename2 = "SET_B_2_2.jpg"
                    break
                elif imageInput == 3:
                    filename1 = "SET_B_3_1.jpg"
                    filename2 = "SET_B_3_2.jpg"
                    break
                elif imageInput == 4:
                    filename1 = "SET_B_4_1.jpg"
                    filename2 = "SET_B_4_2.jpg"
                    break
                elif imageInput == 5:
                    filename1 = "SET_B_5_1.jpg"
                    filename2 = "SET_B_5_2.jpg"
                    break
                elif imageInput == 6:
                    filename1 = "SET_A_1_1.JPG"
                    filename2 = "SET_A_1_2.JPG"
                    break
                elif imageInput == 7:
                    filename1 = "SET_A_2_1.JPG"
                    filename2 = "SET_A_2_2.JPG"
                    break
                elif imageInput == 8:
                    filename1 = "SET_A_3_1.JPG"
                    filename2 = "SET_A_3_2.jpg"
                    break
                elif imageInput == 9:
                    filename1 = "SET_A_4_1.jpg"
                    filename2 = "SET_A_4_2.jpg"
                    break
                elif imageInput == 10:
                    filename1 = "SET_A_5_1.JPG"
                    filename2 = "SET_A_5_2.JPG"
                    break
                else:
                    print("\nExiting program. Goodbye!")
                    sys.exit()
            else:
                print("\nError: Please select a valid option")
        except ValueError:
            print("\nError: Please enter a number")

    img1 = cv2.imread(os.path.join("Resources", filename1), 1)
    img2 = cv2.imread(os.path.join("Resources", filename2), 1)
    imageWidth = max(img1.shape[1], img2.shape[1])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

  # Inside main() function, continuing from the BGR to RGB conversion
    fig, axes = pt.subplots(1, 2)
    axes[0].set_title("Reference Image")
    axes[0].imshow(img1)
    axes[1].set_title("Image to be Aligned")
    axes[1].imshow(img2)
    fig.show()

    # SURF feature detection parameters
    surfHessianThreshold = 300
    goodMatchPercent = 0.15

    # Perform image alignment
    img1only_overlapped, alignedImg_overlapped = imageAlignment(img1, img2, surfHessianThreshold, goodMatchPercent)

    # Calculate window size for SSIM
    winSize = int(0.0017 * imageWidth + 9.7568)
    if winSize % 2 == 0:
        winSize += 1
    winSize = (winSize, winSize)

    # Display overlapped regions
    pt.figure()
    pt.subplot(1, 2, 1)
    pt.title("Reference Image - Overlapped Region Only")
    pt.imshow(img1only_overlapped)
    pt.subplot(1, 2, 2)
    pt.title("Aligned Image - Overlapped Region Only")
    pt.imshow(alignedImg_overlapped)
    pt.show()

    # Calculate SSIM
    ssim_score, SSIMimg = SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize)
    print("\nStructural Similarity Index: ", ssim_score)

    if userInput == 2:
        # Handle image resizing based on user selection
        if resizeInput == 2:
            img1only_overlapped = ResizeWithAspectRatio(img1only_overlapped, 
                                                       img1only_overlapped.shape[1] // resizeFactor, 
                                                       img1only_overlapped.shape[0] // resizeFactor)
            alignedImg_overlapped = ResizeWithAspectRatio(alignedImg_overlapped, 
                                                         alignedImg_overlapped.shape[1] // resizeFactor, 
                                                         alignedImg_overlapped.shape[0] // resizeFactor)
            _, SSIMimg = SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize)
        
        if resizeInput == 3:
            SSIMimg = ResizeWithAspectRatio(SSIMimg, 
                                          SSIMimg.shape[1] // resizeFactor, 
                                          SSIMimg.shape[0] // resizeFactor)
            alignedImg_overlapped = ResizeWithAspectRatio(alignedImg_overlapped, 
                                                         alignedImg_overlapped.shape[1] // resizeFactor, 
                                                         alignedImg_overlapped.shape[0] // resizeFactor)

        # Process and display the differences
        output = processDifferences(img1only_overlapped, alignedImg_overlapped, imageWidth, SSIMimg, resizeFactor, winSize)
        
        fig, axes = pt.subplots(1, 2)
        axes[0].set_title("Reference Image")
        axes[0].imshow(img1only_overlapped)
        axes[1].set_title("Detected Changes (Red Outlines)")
        axes[1].imshow(output)
        fig.show()

if __name__ == "__main__":
    main()