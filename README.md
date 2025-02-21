# Handwritten Prescription Recognition using Visual Transformers

## Description
This project aims to detect and recognize handwritten text on medical prescriptions using Visual Transformers instead of traditional Convolutional Neural Networks (CNNs). The model processes images of handwritten prescriptions and outputs the extracted text, which can help reduce human errors in interpreting medical information.

Key Features:
- Utilizes Visual Transformers for advanced image recognition and text extraction.
- Processes images of medical prescriptions and converts handwritten text into digital text.
- Capable of recognizing medicine names, dosages, and other handwritten details.

---

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/handwritten-prescription-recognition.git
    cd handwritten-prescription-recognition
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    Required libraries include:
    - `pandas`
    - `numpy`
    - `tensorflow`
    - `transformers` (Hugging Face for Visual Transformers)
    - `opencv-python`

3. Ensure the dataset is organized as follows:
    ```
    dataset/
    ├── Training/
    │   ├── training_labels.csv
    │   └── training_images/   # Contains .png or .jpg images
    ├── Testing/
    │   ├── testing_labels.csv
    │   └── testing_images/    # Contains .png or .jpg images
    └── Validation/
        ├── validation_labels.csv
        └── validation_images/ # Contains .png or .jpg images
    ```

---

## Usage
### Data Preprocessing
To preprocess and load the data, run:
```sh
python data.py

To train the model, run:
python model.py
