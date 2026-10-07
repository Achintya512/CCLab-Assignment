import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model_pipeline import train_and_evaluate
from simulator import simulate_provisioning

st.set_page_config(page_title="Cloud VM Resource Forecaster", layout="wide")

st.title("Cloud VM Resource Forecasting & Proactive Provisioning Dashboard")

# Define multiple real-world cloud workload datasets from Numenta Anomaly Benchmark (NAB)
DATASETS = {
    "AWS EC2 Instance 1 (General Compute)": "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "AWS EC2 Instance 2 (Bursting Workload)": "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "AWS RDS Database Server 1": "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "AWS RDS Database Server 2": "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
}

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # The NAB datasets have 'timestamp' and 'value'. We map 'value' to 'cpu_usage'
    df.rename(columns={'value': 'cpu_usage'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

st.sidebar.header("1. Dataset Selection")
selected_dataset_name = st.sidebar.selectbox("Choose a Workload Trace", list(DATASETS.keys()))
df = load_data(DATASETS[selected_dataset_name])

st.sidebar.header("2. Simulation Settings")
vm_capacity = st.sidebar.number_input("Single VM Capacity (CPU Units)", min_value=10, max_value=200, value=20)
safety_margin = st.sidebar.slider("Safety Margin (%)", 0, 50, 10) / 100.0

st.header("1. Historical Workload Data")
st.markdown(f"Visualizing **{selected_dataset_name}** - a real CPU utilization trace from AWS CloudWatch.")
fig_hist = px.line(df.iloc[-2000:], x='timestamp', y='cpu_usage', 
                   title=f"Historical CPU Usage (Last 2000 intervals)")
st.plotly_chart(fig_hist, use_container_width=True)

st.header("2. Model Training & Comparison")
st.markdown("Comparing Linear Regression, Random Forest, and Gradient Boosting to predict future CPU requirements.")

with st.spinner("Training models on the selected dataset..."):
    results, predictions, y_test, timestamps_test = train_and_evaluate(df, target_col='cpu_usage')

results_df = pd.DataFrame(results).T
st.dataframe(results_df.style.highlight_min(axis=0, color='lightgreen'))

best_model = results_df['RMSE'].idxmin()
st.success(f"Best performing model based on RMSE: **{best_model}**")

selected_model = st.sidebar.selectbox("Select Model for Simulation", list(results.keys()), 
                                      index=list(results.keys()).index(best_model))

st.header("3. Prediction vs Actual")
pred_df = pd.DataFrame({
    'timestamp': timestamps_test,
    'Actual': y_test.values,
    'Predicted': predictions[selected_model]
})

fig_pred = go.Figure()
# Show last 500 predictions for better visibility
fig_pred.add_trace(go.Scatter(x=pred_df['timestamp'][-500:], y=pred_df['Actual'][-500:], name='Actual'))
fig_pred.add_trace(go.Scatter(x=pred_df['timestamp'][-500:], y=pred_df['Predicted'][-500:], name='Predicted'))
fig_pred.update_layout(title="Short-term Forecast Accuracy")
st.plotly_chart(fig_pred, use_container_width=True)

st.header("4. Proactive Provisioning Simulation")
st.markdown("Using predictions to proactively allocate virtual machines before the actual demand occurs.")

sim_df = simulate_provisioning(pred_df['Actual'], pred_df['Predicted'], vm_capacity, safety_margin)
sim_df['timestamp'] = pred_df['timestamp'].values

col1, col2, col3 = st.columns(3)
col1.metric("Average Resource Utilization", f"{sim_df['utilization'].mean() * 100:.2f}%")
col2.metric("SLA Violations (Under-provisioned)", f"{sim_df['sla_violation'].sum()} events")
col3.metric("Total VMs Provisioned (Mean)", f"{sim_df['vms_provisioned'].mean():.2f}")

st.subheader("Capacity vs Actual Usage")
fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(x=sim_df['timestamp'][-500:], y=sim_df['actual_usage'][-500:], name='Actual Usage', line=dict(color='orange')))
fig_sim.add_trace(go.Scatter(x=sim_df['timestamp'][-500:], y=sim_df['provisioned_capacity'][-500:], name='Provisioned Capacity', line=dict(dash='dash', color='green')))
st.plotly_chart(fig_sim, use_container_width=True)

st.subheader("Resource Utilization Over Time")
fig_util = px.line(sim_df[-500:], x='timestamp', y='utilization', title="Utilization (%)")
fig_util.add_hline(y=1.0, line_dash="dot", line_color="red", annotation_text="100% Util (SLA Violation Risk)")
st.plotly_chart(fig_util, use_container_width=True)
