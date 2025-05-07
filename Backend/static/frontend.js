// This file contains additional JavaScript functions for enhancing the dashboard

// Format percentage values
function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

// Format currency values
function formatCurrency(value) {
    return '$' + value.toFixed(2).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

// Toggle between different time periods
function setTimeRange(range) {
    // This would filter the data based on the selected time range
    // range could be '1m', '3m', '6m', '1y', 'all'
    console.log('Setting time range to:', range);
    // Then refresh charts with the filtered data
}

// Export data to CSV
function exportToCSV(data, filename) {
    const csvContent = "data:text/csv;charset=utf-8," 
        + data.map(row => row.join(",")).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Update dashboard with real-time simulated data
function enableLiveUpdates() {
    // This would simulate "live" updates for demonstration
    setInterval(() => {
        // Generate random fluctuations
        const fluctuation = (Math.random() - 0.5) * 0.01;
        
        // Get current value
        const currentValue = parseFloat(document.getElementById('portfolioValue').textContent);
        
        // Update with small random change
        const newValue = currentValue * (1 + fluctuation);
        document.getElementById('portfolioValue').textContent = formatCurrency(newValue);
        
        // Maybe also update a "last updated" timestamp
        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
    }, 5000); // Update every 5 seconds
}
// Model selection and execution
document.addEventListener('DOMContentLoaded', function() {
    // Set up model selector event listener
    const modelSelector = document.getElementById('modelSelector');
    const mvoDescription = document.getElementById('mvoDescription');
    const deeprlDescription = document.getElementById('deeprlDescription');
    const riskAversionGroup = document.getElementById('riskAversionGroup');
    
    // Update descriptions when model selection changes
    modelSelector.addEventListener('change', function() {
        if (this.value === 'mvo') {
            mvoDescription.style.display = 'block';
            deeprlDescription.style.display = 'none';
            riskAversionGroup.style.display = 'block';
        } else {
            mvoDescription.style.display = 'none';
            deeprlDescription.style.display = 'block';
            // Risk aversion is also used for DeepRL, so keep it visible
            riskAversionGroup.style.display = 'block';
        }
    });
    
    // Update risk aversion value display when slider changes
    const riskAversionSlider = document.getElementById('riskAversionSlider');
    const riskAversionValue = document.getElementById('riskAversionValue');
    
    riskAversionSlider.addEventListener('input', function() {
        riskAversionValue.textContent = this.value;
    });
});

// Run the selected model
// In static/frontend.js
async function runSelectedModel() {
    const modelSelector = document.getElementById('modelSelector');
    const riskAversionSlider = document.getElementById('riskAversionSlider');
    const modelProgress = document.getElementById('modelProgress');
    const runModelBtn = document.getElementById('runModelBtn');
    
    // Get selected model and risk aversion
    const selectedModel = modelSelector.value;
    const riskAversion = parseFloat(riskAversionSlider.value);
    
    // Show progress and disable button
    modelProgress.style.display = 'block';
    modelProgress.innerHTML = `<div class="progress">
        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" 
             style="width: 100%" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">
             Running ${selectedModel.toUpperCase()} model with risk aversion ${riskAversion}...
        </div>
    </div>
    <div class="text-center mt-2 small text-muted">
        This may take several minutes. Deep RL model can take up to 10 minutes to complete.
    </div>`;
    
    runModelBtn.disabled = true;
    
    try {
        // Determine which endpoint to call
        let endpoint = selectedModel === 'mvo' ? '/api/run_mvo' : '/api/run_deep_rl';
        
        // Call the API
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ risk_aversion: riskAversion })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Show success message and reload data
            showAlert(`${selectedModel.toUpperCase()} model executed successfully!`, 'success');
            // Update UI with new data
            fetchPortfolioMetrics();
            fetchPortfolioAllocations();
            fetchModelComparison();
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    } catch (error) {
        showAlert(`Error running model: ${error.message}`, 'danger');
    } finally {
        // Hide progress indicator and re-enable button
        modelProgress.style.display = 'none';
        runModelBtn.disabled = false;
    }
}

// Display alert messages
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
}