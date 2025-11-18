# Sentiment Analysis Web App (Movie Reviews)  
A machine learning–based application that classifies movie reviews as **Positive** or **Negative** using **NLP, TF-IDF, and Logistic Regression**, deployed with **Streamlit**, featuring secure user authentication and a review history dashboard using SQLite.

---

## Features
- User registration & login with secure password hashing (bcrypt + SQLite)
- Review sentiment prediction (positive or negative)
- Dashboard to view prediction history per user
- TF-IDF text preprocessing and Logistic Regression model
- SQLite database for users & review logs
- Streamlit-based web interface
- Includes Jupyter Notebook for model training (`/training` folder)

---

## Project Structure
Sentiment-Analysis/
│── app.py
│── requirements.txt
│── vectorizer.pkl
│── model.pkl
│── reviews.db
│── Movie_Review.csv # dataset (used only for training)
│── .gitignore
│── training/
│ └── sentiment_training.ipynb

yaml
Copy code

---

## Installation & Setup (Local)

### 1. Clone the Repository

git clone https://github.com/Vikram09-stack/Sentiment-Analysis.git
cd Sentiment-Analysis
2. Create & Activate Virtual Environment
Windows
bash
Copy code
python -m venv venv
venv\Scripts\activate
Mac / Linux
bash
Copy code
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the App
bash
Copy code
streamlit run app.py
Login System
This app includes a real authentication system using SQLite + bcrypt.

Users can create accounts with email/password

Passwords are securely hashed (never stored in plain text)

Each user maintains their own prediction history

No demo login or insecure authentication

UI & Dashboard (Screenshots)
Login Page
<img width="2256" height="1145" alt="Login Screenshot" src="https://github.com/user-attachments/assets/eadc934b-63e2-41d6-a415-59e7cd9d4e2c" />
Sentiment Analyzer
<img width="2363" height="1304" alt="Analyzer Screenshot" src="https://github.com/user-attachments/assets/44356a64-9055-488d-a6b9-ca2121aff7c1" />
Dashboard & History
<img width="2408" height="1365" alt="Dashboard Screenshot" src="https://github.com/user-attachments/assets/5cc43dcf-39c0-48ee-a48e-906b7d572dd9" />
Model Training
The model was trained in a Jupyter Notebook located at:

bash
Copy code
/training/sentiment_training.ipynb
Training workflow includes:

Cleaning dataset

TF-IDF vectorization

Logistic Regression training

Generating evaluation metrics

Exporting model.pkl and vectorizer.pkl

Tech Stack
Component	Technology
UI	Streamlit
ML Model	Logistic Regression (scikit-learn)
NLP	TF-IDF + NLTK stopwords
Database	SQLite
Authentication	bcrypt password hashing
Backend	Python
Deployment	Local / Streamlit Cloud / etc

Future Enhancements
Deployment on Streamlit Cloud / Render / Railway

Email verification

Password reset system

Support for additional datasets

Switch to PostgreSQL or Supabase

Improved UI using Streamlit Components or Tailwind CSS
