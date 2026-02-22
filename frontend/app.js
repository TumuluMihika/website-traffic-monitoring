let chart;
let anomalyDetails = {};

Chart.register(window['chartjs-plugin-zoom']);

function showChart() {
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("chartArea").style.display = "block";
}

function renderChart(data) {

    const ctx = document.getElementById("chart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.historical_dates.concat(data.forecast_dates),
            datasets: [
                {
                    label: "Historical",
                    data: data.historical_values.concat(
                        new Array(data.forecast_dates.length).fill(null)
                    ),
                    borderColor: "#7f8c8d",
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: "Forecast",
                    data: new Array(data.historical_dates.length).fill(null)
                        .concat(data.forecast_values),
                    borderColor: "#2980b9",
                    borderWidth: 3,
                    borderDash: [6, 6],
                    tension: 0.3,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "nearest",
                intersect: true
            },
            plugins: {
                legend: { position: "top" },
                zoom: {
                    pan: { enabled: true, mode: "x" },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: "x"
                    }
                }
            }
        }
    });

    showChart();
}

function plotAnomalies(data) {

    const anomalyData = new Array(chart.data.labels.length).fill(null);
    const pointColors = new Array(chart.data.labels.length).fill(null);

    anomalyDetails = {};

    data.anomalies.forEach((anomaly) => {

        const labelIndex = chart.data.labels.indexOf(anomaly.date);

        if (labelIndex !== -1) {
            anomalyData[labelIndex] = anomaly.actual;

            // 🎨 Severity-Based Colors
            if (anomaly.severity === "Low") {
                pointColors[labelIndex] = "#2ecc71"; // Green
            } 
            else if (anomaly.severity === "Medium") {
                pointColors[labelIndex] = "#f39c12"; // Orange
            } 
            else {
                pointColors[labelIndex] = "#e74c3c"; // Red
            }

            anomalyDetails[labelIndex] = anomaly;
        }
    });

    chart.data.datasets.push({
        label: "Anomalies",
        data: anomalyData,
        backgroundColor: pointColors,
        borderColor: pointColors,
        pointRadius: 8,
        showLine: false
    });

    chart.options.onClick = function(evt, elements) {
        if (elements.length > 0) {
            const index = elements[0].index;

            if (anomalyDetails[index]) {
                const a = anomalyDetails[index];

                Swal.fire({
                    title: "Anomaly Detected",
                    html: `
                        <b>Date:</b> ${a.date}<br>
                        <b>Actual:</b> ${a.actual}<br>
                        <b>Predicted:</b> ${a.predicted}<br>
                        <b>Residual:</b> ${a.residual}<br>
                        <b>Severity:</b> ${a.severity}
                    `,
                    icon: a.severity === "High" ? "error" :
                          a.severity === "Medium" ? "warning" : "info"
                });
            }
        }
    };

    chart.update();
}

async function uploadAndDetect() {

    const fileInput = document.getElementById("fileInput");

    if (!fileInput.files.length) {
        Swal.fire({
            icon: "warning",
            title: "Please upload a CSV file."
        });
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const response = await fetch("http://127.0.0.1:8000/upload-anomaly", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    renderChart(data);
    plotAnomalies(data);

    document.getElementById("anomalyCount").innerText = data.anomaly_count;
}