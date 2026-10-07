import pandas as pd
import numpy as np

def simulate_provisioning(actual_usage, predicted_usage, vm_capacity=20, safety_margin=0.1):
    """
    Simulate proactive VM provisioning based on predictions.
    actual_usage: pd.Series of actual resource usage.
    predicted_usage: np.array of predicted resource usage.
    vm_capacity: Amount of resource one VM can handle.
    safety_margin: Extra buffer added to prediction (0.1 = 10%).
    """
    # Calculate target capacity needed based on prediction + margin
    target_capacity = predicted_usage * (1 + safety_margin)

    # Number of VMs to provision
    vms_provisioned = np.ceil(target_capacity / vm_capacity)
    total_provisioned_capacity = vms_provisioned * vm_capacity

    # Calculate utilization
    # Prevent division by zero
    utilization = np.where(total_provisioned_capacity > 0,
                           actual_usage / total_provisioned_capacity,
                           1.0)

    # Cap utilization calculation at 100% for reporting normal operations,
    # but track SLA violations separately
    
    # SLA Violations: when actual usage exceeds provisioned capacity
    sla_violations = actual_usage > total_provisioned_capacity

    return pd.DataFrame({
        'actual_usage': actual_usage.values,
        'predicted_usage': predicted_usage,
        'vms_provisioned': vms_provisioned,
        'provisioned_capacity': total_provisioned_capacity,
        'utilization': utilization,
        'sla_violation': sla_violations
    })
