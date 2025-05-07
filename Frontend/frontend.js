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