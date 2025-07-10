from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
import traceback
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder="static", template_folder="templates")

class AIFitnessRecommendationSystem:
    def __init__(self):
        self.workout_model = None
        self.diet_model = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def create_synthetic_user_data(self, n_samples=2000):
        """Create synthetic user data for training"""
        np.random.seed(42)
        
        # Generate user profiles
        ages = np.random.normal(35, 12, n_samples).astype(int)
        ages = np.clip(ages, 18, 70)
        
        heights = np.random.normal(170, 10, n_samples)
        heights = np.clip(heights, 150, 200)
        
        weights = np.random.normal(70, 15, n_samples)
        weights = np.clip(weights, 45, 120)
        
        workout_freq = np.random.randint(0, 8, n_samples)
        
        # Calculate BMI
        bmi = weights / (heights / 100) ** 2
        
        # Create realistic correlations
        activity_levels = np.where(workout_freq <= 2, 'Low', 
                         np.where(workout_freq <= 4, 'Medium', 'High'))
        
        # Generate fitness goals based on BMI and activity
        fitness_goals = []
        for i in range(n_samples):
            if bmi[i] < 18.5:
                goal = np.random.choice(['Muscle Gain', 'Maintain'], p=[0.7, 0.3])
            elif bmi[i] > 25:
                goal = np.random.choice(['Weight Loss', 'Maintain'], p=[0.8, 0.2])
            else:
                goal = np.random.choice(['Muscle Gain', 'Weight Loss', 'Maintain'], p=[0.4, 0.3, 0.3])
            fitness_goals.append(goal)
        
        # Generate experience levels
        experience_levels = []
        for freq in workout_freq:
            if freq <= 2:
                exp = np.random.choice(['Beginner', 'Intermediate'], p=[0.8, 0.2])
            elif freq <= 4:
                exp = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], p=[0.3, 0.6, 0.1])
            else:
                exp = np.random.choice(['Intermediate', 'Advanced'], p=[0.4, 0.6])
            experience_levels.append(exp)
        
        return pd.DataFrame({
            'Age': ages,
            'Height': heights,
            'Weight': weights,
            'BMI': bmi,
            'WorkoutFrequency': workout_freq,
            'ActivityLevel': activity_levels,
            'FitnessGoal': fitness_goals,
            'ExperienceLevel': experience_levels
        })
    
    def preprocess_data(self, df):
        """Preprocess data for ML models"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['ActivityLevel', 'FitnessGoal', 'ExperienceLevel']
        for col in categorical_cols:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def train_clustering_model(self, user_data):
        """Train clustering model to group similar users"""
        features = ['Age', 'BMI', 'WorkoutFrequency']
        X = user_data[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use 4 clusters as optimal
        self.clustering_model = KMeans(n_clusters=4, random_state=42)
        self.clustering_model.fit(X_scaled)
        
        return self.clustering_model.labels_
    
    def train_models(self):
        """Train all AI models"""
        print("Training AI models...")
        
        # Create synthetic data
        user_data = self.create_synthetic_user_data()
        
        # Preprocess data
        processed_data = self.preprocess_data(user_data)
        
        # Train clustering
        clusters = self.train_clustering_model(processed_data)
        processed_data['Cluster'] = clusters
        
        # Prepare features for supervised learning
        feature_cols = ['Age', 'BMI', 'WorkoutFrequency', 'ActivityLevel_encoded', 'Cluster']
        X = processed_data[feature_cols]
        
        # Train workout recommendation model
        y_workout = processed_data['ExperienceLevel_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y_workout, test_size=0.2, random_state=42)
        
        self.workout_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.workout_model.fit(X_train, y_train)
        
        # Train diet recommendation model (predict calorie needs)
        # Calculate daily calorie needs based on BMR and activity
        bmr = processed_data.apply(lambda row: self.calculate_bmr(row['Weight'], row['Height'], row['Age']), axis=1)
        activity_multiplier = processed_data['ActivityLevel'].map({'Low': 1.2, 'Medium': 1.55, 'High': 1.725})
        daily_calories = bmr * activity_multiplier
        
        # Adjust calories based on fitness goal
        goal_adjustment = processed_data['FitnessGoal'].map({
            'Weight Loss': 0.8, 'Maintain': 1.0, 'Muscle Gain': 1.2
        })
        target_calories = daily_calories * goal_adjustment
        
        self.diet_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.diet_model.fit(X_train, target_calories.iloc[X_train.index])
        
        # Evaluate models
        workout_pred = self.workout_model.predict(X_test)
        workout_accuracy = accuracy_score(y_test, workout_pred)
        
        diet_pred = self.diet_model.predict(X_test)
        diet_mse = mean_squared_error(target_calories.iloc[X_test.index], diet_pred)
        
        print(f"Workout Model Accuracy: {workout_accuracy:.3f}")
        print(f"Diet Model MSE: {diet_mse:.3f}")
        
        self.is_trained = True
        self.save_models()
    
    def calculate_bmr(self, weight, height, age, gender='Male'):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
        if gender == 'Male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        return bmr
    
    def predict_user_profile(self, user_data):
        """Predict user profile using trained models"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare user data
        user_df = pd.DataFrame([user_data])
        user_df['BMI'] = user_df['Weight'] / (user_df['Height'] / 100) ** 2
        
        # Determine activity level
        freq = user_df['WorkoutFrequency'].iloc[0]
        if freq <= 2:
            activity_level = 'Low'
        elif freq <= 4:
            activity_level = 'Medium'
        else:
            activity_level = 'High'
        
        user_df['ActivityLevel'] = activity_level
        
        # Get cluster
        user_features = user_df[['Age', 'BMI', 'WorkoutFrequency']]
        user_scaled = self.scaler.transform(user_features)
        cluster = self.clustering_model.predict(user_scaled)[0]
        
        # Encode categorical variables
        if 'ActivityLevel' not in self.label_encoders:
            self.label_encoders['ActivityLevel'] = LabelEncoder()
            self.label_encoders['ActivityLevel'].fit(['Low', 'Medium', 'High'])
        
        activity_encoded = self.label_encoders['ActivityLevel'].transform([activity_level])[0]
        
        # Prepare features for prediction
        prediction_features = np.array([[
            user_df['Age'].iloc[0],
            user_df['BMI'].iloc[0],
            user_df['WorkoutFrequency'].iloc[0],
            activity_encoded,
            cluster
        ]])
        
        # Predict experience level
        experience_pred = self.workout_model.predict(prediction_features)[0]
        experience_levels = ['Beginner', 'Intermediate', 'Advanced']
        experience_level = experience_levels[min(experience_pred, len(experience_levels)-1)]
        
        # Predict daily calories
        daily_calories = self.diet_model.predict(prediction_features)[0]
        
        return {
            'cluster': int(cluster),
            'experience_level': experience_level,
            'activity_level': activity_level,
            'daily_calories': int(daily_calories),
            'bmi': float(user_df['BMI'].iloc[0])
        }
    
    def save_models(self):
        """Save trained models"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        joblib.dump(self.workout_model, 'models/workout_model.pkl')
        joblib.dump(self.diet_model, 'models/diet_model.pkl')
        joblib.dump(self.clustering_model, 'models/clustering_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.workout_model = joblib.load('models/workout_model.pkl')
            self.diet_model = joblib.load('models/diet_model.pkl')
            self.clustering_model = joblib.load('models/clustering_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            self.is_trained = True
            print("Models loaded successfully")
        except FileNotFoundError:
            print("No pre-trained models found. Training new models...")
            self.train_models()

# Initialize AI system
ai_system = AIFitnessRecommendationSystem()

# Load datasets
try:
    workout_df = pd.read_csv("datasets/workout_dataset.csv")
    diet_df = pd.read_csv("datasets/diet_dataset.csv")
except FileNotFoundError:
    print("Dataset files not found. Please ensure workout_dataset.csv and diet_dataset.csv are in the datasets folder.")
    workout_df = pd.DataFrame()
    diet_df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['Height', 'Weight', 'Age', 'WorkoutFrequency']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required."}), 400
        
        # Get AI predictions
        user_profile = ai_system.predict_user_profile(data)
        
        # Determine fitness goal based on BMI
        bmi = user_profile['bmi']
        if bmi < 18.5:
            fitness_goal = 'Muscle Gain'
        elif bmi > 25:
            fitness_goal = 'Weight Loss'
        else:
            fitness_goal = 'Maintain'
        
        # Get personalized workout recommendations
        workouts = get_ai_workout_recommendations(
            user_profile['experience_level'], 
            fitness_goal, 
            user_profile['activity_level']
        )
        
        # Get personalized diet recommendations
        diet_recommendations = get_ai_diet_recommendations(
            fitness_goal, 
            user_profile['daily_calories']
        )
        
        # Calculate success probability based on multiple factors
        success_prob = calculate_success_probability(data, user_profile)
        
        return jsonify({
            'cluster': user_profile['cluster'],
            'category': user_profile['experience_level'],
            'activity_level': user_profile['activity_level'],
            'fitness_goal': fitness_goal,
            'daily_calories': user_profile['daily_calories'],
            'bmi': round(user_profile['bmi'], 1),
            'goal_prob': success_prob,
            'workouts': workouts,
            'diet': diet_recommendations
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def get_ai_workout_recommendations(experience_level, fitness_goal, activity_level):
    """Get AI-powered workout recommendations"""
    if workout_df.empty:
        return generate_default_workouts(experience_level, fitness_goal)
    
    # Filter workouts based on AI predictions
    filtered_workouts = workout_df[
        (workout_df['fitness_level'] == experience_level) | 
        (workout_df['goal'] == fitness_goal)
    ]
    
    if filtered_workouts.empty:
        filtered_workouts = workout_df
    
    # Select diverse workouts
    selected_workouts = filtered_workouts.sample(n=min(5, len(filtered_workouts)))
    
    workouts = []
    for _, row in selected_workouts.iterrows():
        workout = {
            'Name': row.get('exercise', 'Custom Workout'),
            'Target': row.get('goal', fitness_goal),
            'Difficulty': row.get('fitness_level', experience_level),
            'Sets': int(row.get('sets', 3)),
            'Reps': int(row.get('reps', 10)),
            'Duration': int(row.get('duration', 5)),
            'Description': f"{row.get('sets', 3)} sets of {row.get('reps', 10)} reps - {row.get('duration', 5)} min"
        }
        workouts.append(workout)
    
    return workouts

def get_ai_diet_recommendations(fitness_goal, daily_calories):
    """Get AI-powered diet recommendations"""
    if diet_df.empty:
        return generate_default_diet(fitness_goal, daily_calories)
    
    # Filter diet based on fitness goal
    filtered_diet = diet_df[diet_df['goal'] == fitness_goal]
    
    if filtered_diet.empty:
        filtered_diet = diet_df
    
    # Select balanced meals
    meal_types = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
    selected_meals = []
    
    for meal_type in meal_types:
        meal_options = filtered_diet[filtered_diet['meal_type'] == meal_type]
        if not meal_options.empty:
            selected_meals.append(meal_options.sample(n=1).iloc[0])
    
    # If not enough meals, fill with random selections
    while len(selected_meals) < 4:
        selected_meals.append(filtered_diet.sample(n=1).iloc[0])
    
    diet_recommendations = []
    for meal in selected_meals:
        diet_item = {
            'Name': meal.get('description', 'Healthy Meal'),
            'MealType': meal.get('meal_type', 'Meal'),
            'Calories': int(meal.get('calories', 300)),
            'Protein': float(meal.get('proteins', 20)),
            'Carbs': float(meal.get('carbs', 30)),
            'Fat': float(meal.get('fats', 10))
        }
        diet_recommendations.append(diet_item)
    
    return diet_recommendations

def calculate_success_probability(user_data, user_profile):
    """Calculate success probability using multiple factors"""
    base_prob = 0.5
    
    # Factor 1: Workout frequency
    freq = user_data.get('WorkoutFrequency', 0)
    freq_factor = min(freq / 5.0, 1.0) * 0.3
    
    # Factor 2: Age factor (younger people tend to see faster results)
    age = user_data.get('Age', 30)
    age_factor = max(0, (50 - age) / 50) * 0.2
    
    # Factor 3: BMI factor (people closer to normal BMI have higher success)
    bmi = user_profile['bmi']
    bmi_factor = max(0, 1 - abs(bmi - 22) / 10) * 0.2
    
    # Factor 4: Activity level
    activity_levels = {'Low': 0.1, 'Medium': 0.15, 'High': 0.2}
    activity_factor = activity_levels.get(user_profile['activity_level'], 0.1)
    
    total_prob = base_prob + freq_factor + age_factor + bmi_factor + activity_factor
    return min(total_prob, 1.0)

def generate_default_workouts(experience_level, fitness_goal):
    """Generate default workouts when dataset is not available"""
    workouts = [
        {
            'Name': 'Push-ups',
            'Target': fitness_goal,
            'Difficulty': experience_level,
            'Sets': 3,
            'Reps': 10 if experience_level == 'Beginner' else 15,
            'Duration': 5,
            'Description': f"3 sets of {10 if experience_level == 'Beginner' else 15} reps - 5 min"
        },
        {
            'Name': 'Squats',
            'Target': fitness_goal,
            'Difficulty': experience_level,
            'Sets': 3,
            'Reps': 12 if experience_level == 'Beginner' else 20,
            'Duration': 5,
            'Description': f"3 sets of {12 if experience_level == 'Beginner' else 20} reps - 5 min"
        },
        {
            'Name': 'Plank',
            'Target': fitness_goal,
            'Difficulty': experience_level,
            'Sets': 3,
            'Reps': 0,
            'Duration': 30 if experience_level == 'Beginner' else 60,
            'Description': f"3 sets of {30 if experience_level == 'Beginner' else 60} seconds"
        }
    ]
    return workouts

def generate_default_diet(fitness_goal, daily_calories):
    """Generate default diet when dataset is not available"""
    meal_calories = daily_calories // 4  # Divide among 4 meals
    
    diet = [
        {
            'Name': 'Healthy Breakfast',
            'MealType': 'Breakfast',
            'Calories': meal_calories,
            'Protein': 20,
            'Carbs': 40,
            'Fat': 10
        },
        {
            'Name': 'Nutritious Lunch',
            'MealType': 'Lunch',
            'Calories': meal_calories,
            'Protein': 30,
            'Carbs': 35,
            'Fat': 15
        },
        {
            'Name': 'Balanced Dinner',
            'MealType': 'Dinner',
            'Calories': meal_calories,
            'Protein': 25,
            'Carbs': 30,
            'Fat': 12
        },
        {
            'Name': 'Healthy Snack',
            'MealType': 'Snack',
            'Calories': meal_calories,
            'Protein': 10,
            'Carbs': 20,
            'Fat': 8
        }
    ]
    return diet

if __name__ == '__main__':
    # Initialize AI system
    ai_system.load_models()
    app.run(debug=True)