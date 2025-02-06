import matplotlib.pyplot as plt
import numpy as np

# Path to the file containing Precision-Recall values
results_file = ""

# Lists to store confidence thresholds, precision, and recall values
thresholds = []
precision = []
recall = []

# Read values from the results file
with open(results_file, "r") as f:
    lines = f.readlines()[2:]  # Skip the first two header lines
    for line in lines:
        values = line.strip().split(" | ")
        thresholds.append(float(values[0]))
        precision.append(float(values[1]))
        recall.append(float(values[2]))

# Compute AUPRC (Area Under Precision-Recall Curve) using NumPy
auprc = np.trapz(precision, recall)

# Plot Precision-Recall curve (without markers)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, linestyle="-", color="b", label=f"PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

print(f"AUPRC (Area Under Precision-Recall Curve): {auprc:.4f}")
