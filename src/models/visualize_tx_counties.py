import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ===============================
# CONFUSION MATRIX DATA
# ===============================

cm_harris = np.array([[189, 102],
                      [14, 60]])

cm_bexar = np.array([[210, 57],
                     [45, 53]])

# ===============================
# FUNCTION TO PLOT CONFUSION MATRIX
# ===============================

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(title, fontsize=14)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

# ===============================
# GENERATE BOTH PLOTS
# ===============================

plot_confusion_matrix(cm_harris, "Harris County — Confusion Matrix")
plot_confusion_matrix(cm_bexar, "Bexar County — Confusion Matrix")
