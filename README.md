# Cloud VM Resource Forecasting & Proactive Provisioning

This project is a machine learning application designed to forecast future Virtual Machine (VM) resource requirements in a cloud data center. It utilizes real-world AWS CloudWatch CPU traces (from the Numenta Anomaly Benchmark) to train predictive models and simulate a proactive cloud provisioning environment.

## Features
* **Authentic Data:** Automatically fetches real-world AWS EC2 and RDS workload traces.
* **Machine Learning Pipeline:** Trains Linear Regression, Random Forest, and Gradient Boosting models using time-series lag features.
* **Cloud Simulator:** Calculates optimal VM provisioning based on predictions, balancing Resource Utilization and SLA Violations using a configurable safety margin.
* **Interactive Dashboard:** A Streamlit web interface to visualize historical data, model accuracy, and real-time simulation metrics.

## Running the Application

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/Achintya512/CCLab-Assignment.git
cd CCLab-Assignment
```
*(If you are already in the project directory locally, skip to step 2).*

### 2. Create a Virtual Environment (Optional but recommended)
```powershell
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
This will start a local web server and open the interactive dashboard in your default web browser. The app will automatically download the required datasets on its first run.
