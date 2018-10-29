# CarND-Term1-P5-VehicleDetection
Self-Driving Car Engineer Nanodegree Program: Term 1 Project 5

## Introduction

The goal of this project was to identify and track vehicles around our vehicle as it drives along. This project used image processing techniques with a linear classifier to detect one class of road obstacles, namely “cars”. Similar techniques could be used to detect other classes of road obstacles such as pedestrians, animals, barriers, lanes, etc. in parallel. The image and video pipeline, and required functions defined by me are all included in [Vehicle_Detection_and_Tracking.ipynb](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/Vehicle_Detection_and_Tracking.ipynb). I will be referencing cells from this notebook along the way.

## Load and Visualize Data

The labeled training and test data for this project was sampled from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from [project_video.mp4](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/project_video.mp4) itself. Cell 1 initializes all libraries. Cell 2 imports and displays some characteristics for the data set. There are 8,792 cars and 8,968 non-cars in this data set so it is fairly balanced. Each image has a size of 64x64x3 pixels which is useful information if the raw pixels need to be used. Cell 3 visualizes some images. The non-car images seem to consist of road lanes, barriers, trees, sky and other background items typically seen on roads. It does not consist of road signs presumably because a traffic sign classifier could be trained in parallel, and would form its own set of classes. Also as far as obstacle detection is considered, traffic signs do not fall in the category of obstacles.

## Features Extraction

I used all the following options for feature extraction demonstrated in the lectures:

**1) Raw pixels data (cell 4):**

Using raw pixels data is the easiest feature to extract from an image. An input image is matched against a known sample image for a car. This approach is typically not self-sufficient since cars on the road are likely to be a different color, shape and at a different orientation from the sample images, no matter how big the training data. The *bin_spatial* function (optionally) resizes an image and creates a vector from all its pixels.

**2) Color histogram data (cell 4):**

The *color_hist* function extracts a histogram of pixels for each layer within the color scheme as features. This information removes dependence on the structure or layout of pixels and relies solely on the color distribution. As long as cars of all possible colors are present in the training data, this technique does well to identify them. But it could also falsely identify other objects sharing a similar color distribution as cars.

![features_extraction_image1](/readme_images/features_extraction_image1.jpg)

**3) Histogram of Oriented Images or HOG (cell 5):**

The *get_hog_features* function computes a histogram of grayscaled pixels within a group of pixels across the image using the *skimage.feature hog* function. Each group of pixels tend to have a certain directional gradient magnitude. All groups together represent the shape of the object in the image. This technique helps identify a car by identifying shapes of car features such as tail lamps, windshields, license plates, etc.

![features_extraction_image2](/readme_images/features_extraction_image2.jpg)

Cell 6 contains the *extract_features* function that calls each of the above functions and returns a feature vector that can be used directly for training the classifier. It also converts the input images to a different color space if required. The default RGB color space was detecting lane lines as cars and took too long to train. The YUV and YCrCb color spaces seemed equally matched in terms of accuracy (both above 98%) and time required to train. I settled for the YCrCb classifier since it resulted in a bit lesser number of features which may ease processing. I cropped each training image from 64x64x3 to 32x32x3 before extracting raw pixels as features without much loss of information. I picked 64 histogram bins to represent the range of 0 to 255 pixels in each image layer as a bin number of 32 or lower (in multiples of 2) either resulted in false positives or did not identify the exact location of cars in the test images. Cell 7 lists all hyperparameters used for training and detection. The *extract_features* function returned 8,556 features for both the car and non-car data sets.

## Training and Testing

I used a linear SVM *LinearSVC* from the *sklearn.svm* library as classifier for the training images (cell 9). Cell 8 normalizes shuffles and splits the data into training and test (validation) sets in a 80:20 ratio. **98.79% test accuracy was obtained**.

## Sliding Window Search

Both sliding windows approaches demonstrated in the lectures were tried out.

- The *slide_window function* (cell 11) returns a list of windows within a marked out area over an image.
- The *draw_boxes* function (cell 11) uses this list to draw boxes on to an image.
- The *single_img_features function* (cell 12) is identical to the *extract_features* function except that it extracts features for a single image only, a behavior that would be useful in the video pipeline.
- The *search_windows* function (cell 12) passes each window (a 64x64 image in this case to match the training images size) through the previously trained classifier. If the classifier predicts that it contains enough information to belong to a car, that window is noted. A final list of “hot” windows is returned which are drawn back on to the original image (cell 13).

The HOG sub-sampling function *find_cars* (cell 13) is a more efficient means of achieving the same result as above since it extracts all features at once instead of one 64x64 window at a time. Hence, it is used in the final video pipeline. This function is identical to the one demonstrated in the lectures except that it manages color space conversion internally and takes an additional argument for the requested color space. It returns an image with the “hot” windows drawn over the original image and a list of the bounding boxes. Cell 15 executes this function on the test images successfully.

![sliding_window_search_image](/readme_images/sliding_window_search_image.jpg)

## Removing False Positives and Multiple Detections

Multiple bounding boxes tend to become noisy between frames and are too confusing for a global obstacle detection algorithm to process. Hence, these boxes were combined into a single bounding box using the *add_heat*, *apply_threshold* and *draw_labeled_bboxes* functions (cell 16). These functions are identical to those demonstrated in the lectures.
- The *add_heat* function adds one pixel count for every pixel within a bounding box while the undetected pixels stay black. The more overlapping a set of pixels are within multiple bounding boxes, the brighter they look (more “heat”).
- The *apply_threshold* function checks the number of counts for each bounding box (how “bright”) and eliminates the ones with lesser counts. This is a means of removing false positives which would have lesser bounding boxes (provided the classifier was well trained).
- The *draw_labeled_bboxes* function collects the remaining “bright” pixels and draws a single rectangle around them.
Cell 17 demonstrates all functions.

![removing_false_positives_image](/readme_images/removing_false_positives_image.jpg)

## Image Pipeline

The function *pipeline_image* has recurring instances of the *find_cars* function with increasing *scale* parameters over a varying region of the image. The result was a lot more bounding boxes of different sizes. This helped detect cars when they were big nearby and as they got smaller near the horizon. These recurring attempts also ensured that true positives were detected more often than false positives. The *add_heat* and *apply_threshold* functions then eliminated false positives more reliably. Cell 19 demonstrates the image pipeline.

## Video Pipeline

The image pipeline by itself detected vehicles accurately. But the bounding boxes were jittery between frames because each frame was treated individually. To introduce some form of filtering and reduce jitter, a *pipeline_video* function was created (cell 21). This function is identical to *pipeline_image* but additionally gathers each bounding box from 15 previous frames for filtering (cell 20). When passed sequentially through the *add_heat* function, the older bounding boxes receive more counts. On passing through the *apply_threshold* function with a higher threshold, the resulting bounding box is slightly larger and moves slower over 15 frames than 15 individual boxes, thus reducing jitter while still identifying the whole car.

Cells 22 and 23 used the moviepy library to import [test_video.mp4](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/test_video.mp4) and [project_video.mp4](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/project_video.mp4) respectively, execute the *pipeline_video* function on each frame and generate [test_video_out.mp4](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/test_video_out.mp4) and [project_video_out.mp4](https://github.com/nvphadnis/CarND-Term1-P5-VehicleDetection/blob/master/project_video_out.mp4) respectively. Vehicle detection and tracking is fairly steady and accurate throughout both videos.

## Discussion

The most challenging part of this project for me was to identify the hyperparameters to be used for training and in the final pipeline. The hit-and-trial process took long due to the classifier training time for every iteration followed by running the image pipeline over the test images. I believe they can be tuned better to be robust to more challenging videos. Some vehicles on the other side of the barriers were also detected which is a good thing if there were no barriers but may potentially confuse a global obstacle detection algorithm. Also there is room for improvement in identifying a vehicle sooner; new vehicles entering from the right side of the frame took a few frames before being identified. The training images seemed to be collected in decent lighting conditions hence this classifier would probably fail under dim lighting conditions. More training data as well as better resolution on the input images to the classifier and pipeline would be required to identify vehicles in the dark, especially those with lights turned off or blinking (eg: stranded on the shoulder by a highway).


