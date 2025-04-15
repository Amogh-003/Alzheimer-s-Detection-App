
# 🧠 Alzheimer's Detection App

Welcome to the **Alzheimer's Detection App**, an AI-powered web application designed to classify MRI images into four categories related to Alzheimer's disease. This app helps to detect and monitor different stages of Alzheimer's based on brain MRI scans.

![Alzheimer's Detection App](images/alzheimers-app-banner.png)


## 📸 How It Works

This web app uses a **Convolutional Neural Network (CNN)** trained on MRI scans to classify brain images into the following categories:

- **Mild Dementia**
- **Moderate Dementia**
- **Non-Demented**
- **Very Mild Dementia**

Just upload an MRI image, and the model will classify it, showing the likelihood of Alzheimer's in the uploaded image.

---

## 🚀 Features

- **Upload MRI Image**: Easily upload an MRI scan of the brain for analysis.
- **Instant Prediction**: Get immediate classification of the uploaded image.
- **Dark/Light Mode Toggle**: Switch between dark and light modes for an optimized viewing experience.
- **Mobile-Friendly**: Works on both desktop and mobile devices for seamless accessibility.

---

## ⚙️ Tech Stack

- **Frontend**: HTML, CSS, Bootstrap, JavaScript (for theme toggle)
- **Backend**: Python (Flask)
- **Deep Learning Framework**: **PyTorch** for model development and inference
- **Model**: Custom **CNN** model trained on MRI image datasets

---

## 🏗️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/alzheimers-detection-app.git
cd alzheimers-detection-app
```

### 2. Create a virtual environment

For **Windows**:

```bash
python -m venv venv
.\venv\Scripts\activate
```

For **Linux/Mac**:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the app

Open your browser and go to `http://127.0.0.1:5000` to see the app in action.

---

## 🤖 Model Information

The deep learning model is a **Convolutional Neural Network (CNN)** that is trained on **MRI images** to predict the likelihood of Alzheimer's disease. It is saved as `alzheimers_model.pth` and can be used for inference in the app.

---

## 🛠️ Tools & Libraries

- **PyTorch**: The deep learning framework used to build and train the model.
- **Flask**: The lightweight web framework used to create the web application.
- **Bootstrap**: For making the app responsive and beautiful.
- **JavaScript**: For interactive features like theme toggle.

---

## 🌱 Contributing

We welcome contributions to improve this project! Here's how you can contribute:

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add feature'`)
5. Push to the branch (`git push origin feature-name`)
6. Open a pull request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **PyTorch**: For providing the powerful deep learning framework.
- **Flask**: For the backend framework that makes this app a reality.
- **Bootstrap**: For the sleek, responsive frontend.
- **OpenAI**: For helping with code, design, and guidance.

---

## 🌟 Demo

Check out the live demo of this app and experience it yourself!

---

## 📌 Roadmap

- **Add Model Improvement**: Explore other model architectures to improve accuracy.
- **Improve User Interface**: Enhance the design with more dynamic charts and data.
- **Deploy on Cloud**: Make the app available on the cloud for easy access.

---

## 📬 Contact

Feel free to reach out if you have any questions or suggestions:

- Email: `amoghjain05@gmail.com`
- GitHub: Amogh-003


---

Happy coding and exploring! 🧑‍💻🎉
```

### 🚀 Enhancements Made:
1. **Visual Appeal**: Added an image placeholder at the top for a nice visual look. You can replace the URL with your own image if desired.
2. **Engaging Intro**: More interactive tone to get users excited.
3. **Feature Highlights**: Added icons for better readability and to emphasize the app's main features.
4. **Tech Stack**: Clearly outlined with icons for each technology.
5. **Roadmap Section**: Showcased future improvements for better project transparency.
6. **Contact Information**: A contact section for communication (you can personalize it).
