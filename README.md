# Emotion Detection Using CNN, Transfer Learning and FER-2013 Dataset

Overview
This project aims to detect human emotions using convolutional neural networks (CNN) trained on the FER-2013 dataset. We implemented various deep learning architectures, including custom CNNs, VGG16, and ResNet50v2, to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The project addresses class imbalance through image augmentation and class weights, achieving significant improvements in model robustness and performance.

Data Source
The FER-2013 dataset consists of 48x48 pixel grayscale images of faces, each labeled with one of the seven emotion categories. The dataset is divided into a training set with 28,709 examples and a public test set with 3,589 examples.

Source: FER-2013 Dataset on Kaggle

# Project Steps

1. **Data Analysis and Understanding**
    - Performed exploratory data analysis to understand the distribution and characteristics of the dataset.
    - Visualized class distributions and sample images to identify class imbalances and data quality.
2. **Custom CNN**
    - Designed and iterated on custom CNN architectures to optimize model performance.
    - Experimented with different layer configurations, activation functions, and kernel initializations.
3. **Callbacks**
    - Implemented callbacks such as `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`, and `CSVLogger` to enhance training efficiency and model performance.
4. **Class Weights**
    - Addressed class imbalance by assigning class weights during model training to give more importance to underrepresented classes.
5. **Image Augmentation**
    - Employed image augmentation techniques such as rotation, zoom, and horizontal flip to artificially increase the diversity of the training dataset.
6. **Transfer Learning**
    - Utilized pre-trained models (VGG16, ResNet50v2) for transfer learning to leverage learned features and improve model accuracy.
7. **Model Evaluation**
    - Evaluated model performance using ROC curves, classification reports, and confusion matrices.
    - Achieved a 66% overall accuracy on emotion classification with the final ResNet50v2 model, detailed through precision, recall, and F1-scores across all emotion labels.
8. **Deployment**
    - Deployed the model for real-time emotion detection in live video streams using Gradio and OpenCV.
    - The deployment showcases dynamic emotion labels on-screen, enabling real-time emotion recognition.

# Real-Time Emotion Detection with OpenCV

## Flow Description

1. **Importing Libraries:**
    - Import necessary libraries: OpenCV for computer vision tasks, NumPy for numerical operations, and TensorFlow for deep learning. Additionally, a utility from Keras is imported to handle image-to-array conversions.
2. **Loading the Pre-Trained Model:**
    - Load a pre-trained deep learning model, specifically a ResNet50 model fine-tuned for emotion recognition, from a file.
3. **Defining Emotion Labels:**
    - Define a list of emotion labels corresponding to the output classes of the model, such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
4. **Initializing the Face Classifier:**
    - Load a pre-trained Haar Cascade classifier for detecting faces. This classifier is used for real-time face detection due to its speed and efficiency.
5. **Starting Video Capture:**
    - Initialize video capture using the default camera (typically the webcam) to continuously read frames from the camera in real time.
6. **Processing Each Video Frame:**
    - Convert each frame from its original BGR color space to grayscale, as face detection works more efficiently on grayscale images.
7. **Face Detection:**
    - Apply the face classifier to the grayscale frame to detect faces. The classifier scans the image at multiple scales to identify regions likely containing faces.
8. **Processing Detected Faces:**
    - For each detected face, extract the face region from the grayscale frame, resize it to 224x224 pixels, normalize, convert to an array, and reshape to match the input requirements of the model.
9. **Predicting Emotions:**
    - Feed the pre-processed face into the model to predict the emotion. Select the emotion with the highest probability as the predicted emotion.
10. **Annotating the Frame:**
    - Draw a rectangle around each detected face in the original frame and display the predicted emotion label above the corresponding face rectangle.
11. **Displaying the Annotated Frame:**
    - Display the annotated frame, showing detected faces and their predicted emotions, in a window titled "Emotion Detector".
12. **Handling User Input:**
    - Wait briefly for user input. If the 'q' key is pressed, break the loop and proceed to clean up.
13. **Releasing Resources:**
    - Release the video capture object and close all OpenCV windows to free up resources.
   
## Results
 - Overall Accuracy: 66%
 - Detailed Metrics: Precision, recall, and F1-scores for each of the 7 emotion labels are available in the evaluation reports.


## Acknowledgments
FER-2013 Dataset: Kaggle
Mentor: Balaji Chippada(https://www.linkedin.com/in/balaji-chippada-0317/)
