// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const userForm = document.getElementById('userForm');

    userForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const submitBtn = userForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.textContent;
        submitBtn.textContent = 'Processing...';
        submitBtn.disabled = true;

        try {
            const formData = new FormData(userForm);
            const data = {};

            for (const [key, value] of formData.entries()) {
                data[key] = isNaN(value) ? value : Number(value);
            }

            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            displayResults(result);
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert(`There was an error processing your request: ${error.message}`);
        } finally {
            submitBtn.textContent = originalBtnText;
            submitBtn.disabled = false;
        }
    });

    function displayResults(result) {
        document.getElementById('results').style.display = 'block';
        document.getElementById('cluster').textContent = formatCluster(result.cluster);
        document.getElementById('category').textContent = result.category;
        document.getElementById('goalProb').textContent = formatProbability(result.goal_prob);

        // Show workouts
        const workoutsEl = document.getElementById('workouts');
        workoutsEl.innerHTML = '';
        if (result.workouts && result.workouts.length > 0) {
            result.workouts.forEach(workout => {
                const workoutEl = document.createElement('div');
                workoutEl.className = 'workout-item';

                const name = workout.Name || workout.name || 'Unnamed Workout';
                const muscle = workout.Muscle || workout.muscle || workout.Target || workout.target || '';
                const difficulty = workout.Difficulty || workout.difficulty || '';
                const description = workout.Description || workout.description || '';

                workoutEl.innerHTML = `
                    <h4>${name}</h4>
                    ${muscle ? `<p><strong>Target:</strong> ${muscle}</p>` : ''}
                    ${difficulty ? `<p><strong>Difficulty:</strong> ${difficulty}</p>` : ''}
                    ${description ? `<p>${description}</p>` : ''}
                `;
                workoutsEl.appendChild(workoutEl);
            });
        } else {
            workoutsEl.innerHTML = '<p>No specific workout recommendations available.</p>';
        }

        // Show diet
        const dietEl = document.getElementById('diet');
        dietEl.innerHTML = '';
        if (result.diet && result.diet.length > 0) {
            result.diet.forEach(food => {
                const foodEl = document.createElement('div');
                foodEl.className = 'food-item';

                const name = food.Food || food.Name || food.name || food.food || 'Unnamed Food';
                const calories = food.Calories || food.calories || food.kcal || 0;
                const protein = food.Protein || food.protein || 0;
                const carbs = food.Carbs || food.carbs || food.Carbohydrates || food.carbohydrates || 0;
                const fat = food.Fat || food.fat || 0;

                foodEl.innerHTML = `
                    <h4>${name}</h4>
                    <p><strong>Calories:</strong> ${calories} kcal</p>
                    <div class="food-macros">
                        <span class="macro protein"><strong>Protein:</strong> ${protein}g</span>
                        <span class="macro carbs"><strong>Carbs:</strong> ${carbs}g</span>
                        <span class="macro fat"><strong>Fat:</strong> ${fat}g</span>
                    </div>
                `;
                dietEl.appendChild(foodEl);
            });

            createMacroChart(result.diet);
        } else {
            dietEl.innerHTML = '<p>No specific diet recommendations available.</p>';
        }
    }

    function createMacroChart(dietItems) {
        const macros = dietItems.reduce((acc, curr) => {
            acc.protein += parseFloat(curr.Protein || curr.protein || 0);
            acc.fat += parseFloat(curr.Fat || curr.fat || 0);
            acc.carbs += parseFloat(curr.Carbs || curr.carbs || curr.Carbohydrates || curr.carbohydrates || 0);
            return acc;
        }, { protein: 0, fat: 0, carbs: 0 });

        macros.protein = Math.round(macros.protein * 10) / 10;
        macros.fat = Math.round(macros.fat * 10) / 10;
        macros.carbs = Math.round(macros.carbs * 10) / 10;

        const ctx = document.getElementById('macroChart').getContext('2d');

        if (window.macroChart && typeof window.macroChart.destroy === 'function') {
            window.macroChart.destroy();
        }

        window.macroChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: [
                    `Protein (${macros.protein}g)`,
                    `Fat (${macros.fat}g)`,
                    `Carbs (${macros.carbs}g)`
                ],
                datasets: [{
                    data: [macros.protein, macros.fat, macros.carbs],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' },
                    title: { display: true, text: 'Macronutrient Distribution' }
                }
            }
        });
    }

    function formatCluster(cluster) {
        const clusterNames = ['Beginner', 'Intermediate', 'Advanced'];
        return clusterNames[cluster] || `Group ${cluster}`;
    }

    function formatProbability(prob) {
        const percentage = Math.round(prob * 100);
        return `${percentage}%`;
    }
});
