AI-Powered Science Quiz Generator

📖 Overview

Creating high-quality multiple-choice questions (MCQs) is essential in education but often time-consuming and repetitive. This project solves that by using Natural Language Processing (NLP) to automatically generate context-aware science quizzes from any given text.

👉 Input: A block of scientific text with 4 options
👉 Output: Well-structured MCQs with correct answers and realistic distractors

This showcases how AI + web development can create real-world EdTech tools that benefit educators and students alike.

🛠️ Tech Stack

Python → Core application logic

Hugging Face Transformers → Pre-trained NLP model for question generation

Flask → REST API to serve AI predictions

HTML / CSS / JavaScript (Jinja) → Interactive frontend for quiz creation

🤖 AI Model Workflow

User Input → Paste any scientific passage with 4 options in the inputs

Backend Processing → Flask sends text to Hugging Face model

NLP Model → Extracts key concepts, generates a question, correct answer, and distractors

Output → JSON response rendered as multiple-choice questions in the browser

This pipeline demonstrates end-to-end model deployment — from NLP preprocessing to frontend integration.

⚡ Getting Started
1. Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Create a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the app
python app.py


Then open 👉 http://127.0.0.1:5000

📂 Project Structure
.
├── app.py             # Flask backend + model integration
├── index.html         # Frontend UI
├── static/            # CSS + JavaScript
└── README.md          # Project docs

🎯 Key Highlights

🔹 AI-driven automation → Eliminates manual quiz creation

🔹 Full-stack integration → Flask API + dynamic frontend

🔹 Practical EdTech application → Useful for teachers, tutors, and students

🔹 Deployment-ready → Can be hosted locally or on cloud platforms


💻 Frontend Demo
<img width="946" height="431" alt="project1" src="https://github.com/user-attachments/assets/3c2f43fb-72cf-4a55-a930-780624319ec2" />
<img width="960" height="442" alt="project2" src="https://github.com/user-attachments/assets/ee2cee68-bf19-4b9b-a14f-0d0a48393254" />



📜 License

This project is licensed under the MIT License.

