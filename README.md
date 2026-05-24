# Object Detection Using TensorFlow

A real-time Object Detection project built using **TensorFlow**, **OpenCV**, and **Python**.  
This project detects and recognizes multiple objects from images, videos, or webcam streams using pre-trained TensorFlow models.

---

# 📌 Features

✅ Real-time object detection  
✅ Detects multiple objects simultaneously  
✅ Bounding boxes with confidence scores  
✅ Uses TensorFlow pre-trained models  
✅ Works with images, videos, and webcam  
✅ COCO dataset labels support  
✅ Easy to run and beginner-friendly  

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

---

# 📂 Project Structure

```bash
Object-Detection-Using-Tensorflow/
│
├── object_detection.py
├── object_detection.ipynb
├── coco_labels.txt
├── image.jpg
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sathvik0728/Object-Detection-Using-Tensorflow.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd Object-Detection-Using-Tensorflow
```

---

## 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run the Python file:

```bash
python object_detection.py
```

Or run the Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
object_detection.ipynb
```

---

# 🧠 How It Works

1. Loads TensorFlow pre-trained object detection model  
2. Reads image/video/webcam input  
3. Detects objects using deep learning  
4. Draws bounding boxes around detected objects  
5. Displays object labels with confidence percentage  

---

# 📷 Supported Objects

The project uses COCO labels dataset which can detect:

- Person
- Car
- Bicycle
- Dog
- Cat
- Chair
- Bottle
- Laptop
- Mobile Phone
- And many more...

---

# 📸 Sample Output

The model detects objects and displays:

✅ Object Name  
✅ Confidence Score  
✅ Bounding Box  

Example:

```text
Person - 98%
Car - 95%
Dog - 93%
```

---

# 📦 Requirements

Create a file named:

```bash
requirements.txt
```

Add:

```txt
tensorflow
opencv-python
numpy
matplotlib
```

---

# 🚫 Large Files Notice

Pre-trained TensorFlow model files are very large and are not uploaded to GitHub because GitHub has a file size limit.

Ignored files/folders:

- TensorFlow model folders
- Video files
- Cache files
- Jupyter checkpoints

---

# 📄 .gitignore

```gitignore
# TensorFlow model folders
faster-rcnn-*
mobilenet-*

# Videos
*.mp4

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc

# Virtual environments
venv/
env/
```

---

# 💡 Future Improvements

- Real-time webcam detection
- YOLO integration
- Custom object training
- GUI interface
- Streamlit web app deployment

---

# 👨‍💻 Author

## Sathvik

Computer Science Engineering Student  
Passionate about AI, Machine Learning, and Computer Vision

GitHub:  
https://github.com/Sathvik0728

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
📢 Share with others  

---

# 📜 License

This project is open-source and available under the MIT License.
