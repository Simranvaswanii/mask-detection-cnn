# 😷 Mask Detection using CNN  Real-time face mask detection built with Python, OpenCV, and a CNN. Detects faces from a live webcam feed and classifies them as Mask or No Mask. Includes a simple CRUD interface for data management and WEKA-based preprocessing.  **Run:** `mask.py]`  **Author:** Simran Vaswani
## screenshot
<img width="1920" height="1080" alt="login Page" src="https://github.com/user-attachments/assets/b3eecf20-4d21-4d77-a52a-358fb2768f1d" />
<img width="1920" height="1051" alt="loading page" src="https://github.com/user-attachments/assets/96b0656b-d8f2-47e1-9f09-da1b6f8fbe9c" />
<img width="1916" height="1080" alt="detection page" src="https://github.com/user-attachments/assets/c235339f-87eb-46b3-a7b8-31ac1c2ff2c8" />
<img width="1920" height="1060" alt="detection1 page" src="https://github.com/user-attachments/assets/9be001a7-7e47-4539-9098-7fa7ff58a852" />
Key Modules

Data Collection and Preprocessing: gathers masked/unmasked face images, then resizes, normalizes, and labels them (WEKA was used for preprocessing and classification experiments).
CNN Model Training: builds and trains the Convolutional Neural Network to classify faces as Mask or No Mask.
Face Detection (OpenCV): captures the live webcam feed and locates faces in each frame.
Real-Time Mask Classification: passes each detected face to the CNN and shows a Mask / No Mask label on screen.
CRUD Interface: lets users add, view, update, and delete records for data management.
Blockchain Fundamentals (exploration): hashing, Merkle Trees, and transaction security.
