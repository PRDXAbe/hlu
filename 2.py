
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
data = pd.DataFrame({
    "Time": pd.date_range("2026-01-01", periods=30, freq="min"),
    "Accuracy": np.random.uniform(92, 99, 30),
    "CPU": np.random.uniform(20, 80, 30),
    "Memory": np.random.uniform(30, 90, 30),
    "Latency": np.random.uniform(40, 120, 30),
    "Throughput": np.random.uniform(100, 200, 30)
})
data["Drift"] = abs(data["Accuracy"] - 95)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Accuracy", "CPU & Memory", "Latency", "Drift")
)

for name, row, col in [
    ("Accuracy", 1, 1),
    ("CPU", 1, 2),
    ("Memory", 1, 2),
    ("Drift", 2, 2)
]:
    fig.add_trace(go.Scatter(
        x=data["Time"], y=data[name],
        name=name, mode="lines"
    ), row=row, col=col)

fig.add_trace(go.Bar(
    x=data["Time"], y=data["Latency"], name="Latency"
), row=2, col=1)

fig.update_layout(
    title="Agile Runtime Dashboard",
    template="plotly_dark", height=700
)
fig.show()
print(data.tail())
