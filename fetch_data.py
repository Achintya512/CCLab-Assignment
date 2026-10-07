import pandas as pd

def fetch_workload_data(file_path='workload_data.csv'):
    # Real AWS Cloudwatch EC2 CPU utilization dataset from Numenta Anomaly Benchmark (NAB)
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"
    print(f"Downloading real AWS EC2 CPU utilization trace from {url}...")
    
    df = pd.read_csv(url)
    
    # The NAB dataset has 'timestamp' and 'value'. Let's map 'value' to 'cpu_usage'.
    df.rename(columns={'value': 'cpu_usage'}, inplace=True)
    
    df.to_csv(file_path, index=False)
    print(f"Real data downloaded and saved to {file_path}")

if __name__ == "__main__":
    fetch_workload_data()
