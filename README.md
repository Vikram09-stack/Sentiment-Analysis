# 🎬 Sentiment Analysis Web App (Movie Reviews)  
A machine learning–powered sentiment analyzer that classifies movie reviews as **Positive** or **Negative** using **NLP + Logistic Regression + TF-IDF**, deployed with **Streamlit**, and featuring **real user authentication with SQLite**.



## 🚀 Features
- 🔐 User Registration & Login (Secure hashed passwords using bcrypt + SQL)
- 📊 Dashboard with review history & filtering
- 🤖 Trained ML model (Logistic Regression)
- 🧹 Text preprocessing (stopwords removal, cleaning, TF-IDF)
- 💾 SQLite database for user accounts + review logs
- 🌐 Streamlit UI – fast and lightweight
- 📝 Model training included via Jupyter Notebook (`/training` folder)

---

## 📂 Project Structure
Sentiment-Analysis/
│── app.py
│── requirements.txt
│── vectorizer.pkl
│── model.pkl
│── reviews.db
│── Movie_Review.csv # optional dataset, used for training
│── .gitignore
│── training/
│ └── sentiment_training.ipynb





## 🛠️ Installation & Setup (Local)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Vikram09-stack/Sentiment-Analysis.git
cd Sentiment-Analysis
2️⃣ Create and activate virtual environment
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
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the app
bash
Copy code
streamlit run app.py
🔑 Login System
This app includes a real authentication system using SQLite + bcrypt.

✔ Users can create accounts with email + password
✔ Passwords are stored securely (hashed, never in plain text)
✔ Each user has their own review history
✔ No fake/demo login — real credentials only

📊 UI & Dashboard (Add Screenshots Here)
🔐 Login Page
<img width="2256" height="1145" alt="Screenshot 2025-11-18 102001" src="https://github.com/user-attachments/assets/eadc934b-63e2-41d6-a415-59e7cd9d4e2c" />



🧠 Sentiment Analyzer
<img width="2363" height="1304" alt="image" src="https://github.com/user-attachments/assets/44356a64-9055-488d-a6b9-ca2121aff7c1" />


📈 Dashboard & History
<img width="2408" height="1365" alt="image" src="https://github.com/user-attachments/assets/5cc43dcf-39c0-48ee-a48e-906b7d572dd9" />


📚 Model Training
The model was trained using a Jupyter Notebook located in:

bash
Copy code
/training/sentiment_training.ipynb
Steps include:

Cleaning dataset

Generating TF-IDF vectorizer

Training Logistic Regression

Saving model.pkl & vectorizer.pkl

⚙️ Tech Stack
Component	Technology
UI	Streamlit
ML model	Logistic Regression + Scikit-learn
NLP	NLTK stopwords + TF-IDF
DB	SQLite
Auth	bcrypt password hashing
Backend	Python
Deployment	Local / Streamlit Cloud / etc

💡 Future Enhancements
Deploy to Streamlit Cloud / Render / Railway

Add email verification

Add password reset option

Add support for multiple datasets

Switch to PostgreSQL or Supabase backend

Improve UI using Streamlit Components / TailwindCSS

