document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const predictionForm = document.getElementById('prediction-form');
    const sampleDataBtn = document.getElementById('sample-data-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loader = document.getElementById('loader');
    const resultDisplay = document.getElementById('result-display');
    const resultIcon = document.getElementById('result-icon');
    const resultText = document.getElementById('result-text');
    const probabilityBar = document.getElementById('probability-bar');
    const probabilityValue = document.getElementById('probability-value');
    const importanceChart = document.getElementById('importance-chart');
    
    // Sample data based on dataset averages
    const sampleData = {
        ph: 7.08,
        hardness: 196.37,
        solids: 20775.32,
        chloramines: 7.12,
        sulfate: 333.78,
        conductivity: 426.05,
        organic_carbon: 14.28,
        trihalomethanes: 66.40,
        turbidity: 3.97
    };
    
    // Fill form with sample data
    sampleDataBtn.addEventListener('click', function() {
        for (const [key, value] of Object.entries(sampleData)) {
            document.getElementById(key).value = value;
        }
    });
    
    // Form submission
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loader, hide results
        loader.style.display = 'block';
        resultDisplay.style.display = 'none';
        
        // Get form data
        const formData = {
            ph: document.getElementById('ph').value,
            hardness: document.getElementById('hardness').value,
            solids: document.getElementById('solids').value,
            chloramines: document.getElementById('chloramines').value,
            sulfate: document.getElementById('sulfate').value,
            conductivity: document.getElementById('conductivity').value,
            organic_carbon: document.getElementById('organic_carbon').value,
            trihalomethanes: document.getElementById('trihalomethanes').value,
            turbidity: document.getElementById('turbidity').value
        };
        
        // Send data to backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loader, show results
            loader.style.display = 'none';
            resultDisplay.style.display = 'block';
            
            // Update result display
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loader.style.display = 'none';
            alert('An error occurred while making the prediction. Please try again.');
        });
    });
    
    // Display prediction results
    function displayResults(data) {
        const prediction = data.prediction;
        const probability = data.probability * 100;
        const featureImportances = data.feature_importances;
        
        // Update result icon and text
        if (prediction === 1) {
            resultIcon.innerHTML = '<i class="fas fa-check-circle" style="color: var(--accent-color);"></i>';
            resultText.textContent = 'Potable Water';
            resultText.style.color = 'var(--accent-color)';
        } else {
            resultIcon.innerHTML = '<i class="fas fa-times-circle" style="color: var(--danger-color);"></i>';
            resultText.textContent = 'Non-Potable Water';
            resultText.style.color = 'var(--danger-color)';
        }
        
        // Update probability bar
        probabilityBar.style.width = `${probability}%`;
        probabilityValue.textContent = `${probability.toFixed(2)}%`;
        
        // Update feature importance chart
        importanceChart.innerHTML = '';
        
        // Convert to array, sort by importance
        const importanceArray = Object.entries(featureImportances)
            .map(([feature, importance]) => ({ feature, importance }))
            .sort((a, b) => b.importance - a.importance);
        
        // Get max importance for scaling
        const maxImportance = importanceArray[0].importance;
        
        // Create bars for each feature
        importanceArray.forEach(item => {
            const { feature, importance } = item;
            const percentage = (importance / maxImportance) * 100;
            
            // Format feature name
            let featureName = feature.replace(/_/g, ' ');
            featureName = featureName.charAt(0).toUpperCase() + featureName.slice(1);
            
            const barHTML = `
                <div class="importance-bar">
                    <div class="importance-label">
                        <span class="importance-label-name">${featureName}</span>
                        <span class="importance-label-value">${(importance * 100).toFixed(2)}%</span>
                    </div>
                    <div class="importance-meter">
                        <div class="importance-value" style="width: ${percentage}%;"></div>
                    </div>
                </div>
            `;
            
            importanceChart.innerHTML += barHTML;
        });
    }
    
    // Initialize tooltips
    const tooltips = document.querySelectorAll('.info-tooltip');
    tooltips.forEach(tooltip => {
        const title = tooltip.getAttribute('title');
        tooltip.setAttribute('data-original-title', title);
        
        tooltip.addEventListener('mouseenter', function() {
            this.setAttribute('title', '');
        });
        
        tooltip.addEventListener('mouseleave', function() {
            this.setAttribute('title', this.getAttribute('data-original-title'));
        });
    });
});
