# Facial_Expression_Detect

A real-time facial expression recognition system built with a Convolutional Neural Network (CNN) using TensorFlow and OpenCV. This project detects faces from a webcam feed and classifies their expressions into one of seven categories.

---

### Features: 

* **Real-Time Detection**: Identifies faces and predicts expressions instantly from a webcam feed.
* **7 Core Expressions**: Classifies faces into Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* **CNN Architecture**: Built on a robust Convolutional Neural Network for effective feature extraction.
* **Data Augmentation**: Trained on an augmented dataset to improve model generalization and robustness.

---

### Technologies Used: 

* **Python**: Core programming language.
* **TensorFlow/Keras**: For building and training the CNN model.
* **OpenCV**: For real-time face detection and image processing.
* **NumPy**: For numerical operations and data manipulation.

---

### Install the required dependencies: 

This project's dependencies are listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```
---

### How to use:
The project is divided into two main parts: training the model and running the live demo.

**1. Train the Model (Optional):**
If you want to train the model from scratch on the FER2013 dataset, run the training script. The trained model will be saved in the models/ directory.
```bash
python train.py 
```
**2. Run the live demo:**
To see the real-time expression recognition using your webcam, run the demo script. This script loads the pre-trained model from the models/ folder 
```bash
python run_demo.py
```
Press the 'q' key to exit the application.




