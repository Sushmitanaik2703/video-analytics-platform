import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the comparison results
df = pd.read_csv("features/college_model_comparison.csv")

# Convert labels to binary
y_true = df["Actual"].map({"NonFight": 0, "Fight": 1})

models = {
    "Original": df["Original Prediction"].map({"NonFight": 0, "Fight": 1}),
    "Augmented": df["Augmented Prediction"].map({"NonFight": 0, "Fight": 1})
}

os.makedirs("reports", exist_ok=True)

report_lines = []

for model_name, y_pred in models.items():

    # Calculate metrics manually
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    fight_recall = TP / (TP + FN) if (TP + FN) else 0
    nonfight_recall = TN / (TN + FP) if (TN + FP) else 0
    fight_precision = TP / (TP + FP) if (TP + FP) else 0

    report_lines.append(f"\n===== {model_name} MODEL =====")
    report_lines.append(f"Accuracy: {accuracy:.2%}")
    report_lines.append(f"Fight Precision: {fight_precision:.2%}")
    report_lines.append(f"Fight Recall: {fight_recall:.2%}")
    report_lines.append(f"NonFight Recall: {nonfight_recall:.2%}")
    report_lines.append(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    print("\n" + "=" * 40)
    print(f"{model_name} MODEL")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Fight Precision: {fight_precision:.2%}")
    print(f"Fight Recall: {fight_recall:.2%}")
    print(f"NonFight Recall: {nonfight_recall:.2%}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Save text report
with open("reports/college_evaluation_report.txt", "w") as f:
    f.write("COLLEGE VIDEO EVALUATION REPORT\n")
    f.write("Dataset: 3 staged Fight + 7 NonFight videos\n")
    f.write("Note: Small exploratory test; not a benchmark.\n")
    f.write("\n".join(report_lines))

# Accuracy comparison chart
accuracies = []

for model_name, y_pred in models.items():
    accuracies.append((y_true == y_pred).mean() * 100)

plt.figure(figsize=(7, 5))
plt.bar(["Original", "Augmented"], accuracies)
plt.ylabel("Accuracy (%)")
plt.title("College Video Model Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("reports/college_accuracy_comparison.png")
plt.close()

# Confusion matrices
for model_name, y_pred in models.items():

    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))

    matrix = [[TN, FP], [FN, TP]]

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xticks([0, 1], ["NonFight", "Fight"])
    plt.yticks([0, 1], ["NonFight", "Fight"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i][j], ha="center", va="center", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"reports/{model_name.lower()}_confusion_matrix.png")
    plt.close()

print("\nReports generated successfully!")
print("Folder: reports/")
print("Files:")
print("- college_evaluation_report.txt")
print("- college_accuracy_comparison.png")
print("- original_confusion_matrix.png")
print("- augmented_confusion_matrix.png")