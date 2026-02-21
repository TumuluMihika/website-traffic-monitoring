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

            layout: {
                padding: {
                    right: 0
                }
            },

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
            },

            scales: {
                x: {
                    offset: false,
                    ticks: {
                        autoSkip: true,
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });

    showChart();
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

function plotAnomalies(data) {

    const anomalyData = new Array(chart.data.labels.length).fill(null);
    anomalyDetails = {};

    data.anomaly_dates.forEach((date, index) => {

        const labelIndex = chart.data.labels.indexOf(date);

        if (labelIndex !== -1) {
            anomalyData[labelIndex] = data.anomaly_actual[index];

            anomalyDetails[labelIndex] = {
                date: date,
                actual: data.anomaly_actual[index],
                predicted: data.anomaly_predicted[index],
                residual: data.anomaly_residual[index],
                severity: data.severity[index]
            };
        }
    });

    chart.data.datasets.push({
        label: "Anomalies",
        data: anomalyData,
        borderColor: "#e74c3c",
        backgroundColor: "#e74c3c",
        pointRadius: 8,
        showLine: false
    });

    chart.update();
}