# 🏋️ AI Gym Workout and Diet Recommendation System

A smart AI-powered fitness assistant that recommends personalized **workout plans**, **diet suggestions**, and visualizes **macronutrient breakdowns** — all based on user inputs like height, weight, and workout frequency.

---

## 🚀 Features

✅ Personalized fitness grouping using BMI  
✅ Smart workout & diet recommendations (from dataset)  
✅ Macronutrient pie chart (using Chart.js)  
✅ Clean responsive frontend (HTML + CSS + JS)  
✅ Flask-powered Python backend  
✅ AI-based decision logic using BMI and frequency rules

---

## 📁 Folder Structure

AI_Gym_AI_Project/
├── app.py # Flask backend
├── train_gym_model.py # (Optional) ML model trainer
├── datasets/
│ ├── workout_dataset.csv
│ └── diet_dataset.csv
├── static/
│ ├── script.js
│ └── style.css
├── templates/
│ └── index.html
└── README.md


---

## 💡 How It Works

1. Calculates **BMI** from user input (Height & Weight)
2. Groups user as:
   - `Beginner`: Underweight
   - `Intermediate`: Normal
   - `Advanced`: Overweight
3. Suggests:
   - 3 random workouts based on fitness level
   - 3 random meals from the dataset
4. Displays **success probability** and **macronutrient pie chart**

---

## 🧪 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Gym-Trainer.git
cd AI-Gym-Trainer
```
### 2. Install Dependencies

```bash
pip install flask pandas
```
### 3. Run the App
```bash
python app.py
```
### 4. Visit in Browser
```text
http://127.0.0.1:5000
```
---

### 📚 Datasets Used

1. workout_dataset.csv: Exercises, reps, sets, duration

2. diet_dataset.csv: Meal descriptions, calories, protein, carbs, fats

---

### 🔮 Future Ideas

1. Add a real ML model instead of rule-based logic

2. Integrate a chatbot using OpenAI or local NLP

3. Store user history and visualize progress

---
### 🤝 Contributing

PRs and feedback are welcome! Feel free to fork and improve the system. 💪

---

### 📜 License

This project is licensed under the MIT License.

---
```yml

Let me know if you want:
- A version with screenshots
- Hindi-translated README
- README badge icons and GitHub stats

Otherwise, you're all set to upload this to GitHub. ✅

---


