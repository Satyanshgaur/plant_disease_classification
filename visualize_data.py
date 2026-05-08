import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the dataset
BASE_DIR = os.path.expanduser("~/plant_classification/PlantVillage-Dataset/raw/color")

def collect_data(base_dir):
    data = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        
        if "___" in class_name:
            plant, disease = class_name.split("___")
        else:
            plant, disease = class_name, "Unknown"
            
        is_healthy = "healthy" in disease.lower()
        
        data.append({
            "Class": class_name,
            "Plant": plant,
            "Disease": disease,
            "Count": count,
            "Status": "Healthy" if is_healthy else "Diseased"
        })
    return pd.DataFrame(data)

def plot_data(df):
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Top 15 Classes
    plt.figure(figsize=(12, 8))
    top_15 = df.sort_values("Count", ascending=False).head(15)
    sns.barplot(x="Count", y="Class", data=top_15, palette="viridis")
    plt.title("Top 15 Most Common Plant-Disease Classes")
    plt.tight_layout()
    plt.savefig("top_15_classes.png")
    print("Saved top_15_classes.png")

    # 2. Distribution by Plant
    plt.figure(figsize=(12, 6))
    plant_counts = df.groupby("Plant")["Count"].sum().sort_values(ascending=False)
    plant_counts.plot(kind="bar", color="skyblue")
    plt.title("Number of Images per Plant Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("distribution_by_plant.png")
    print("Saved distribution_by_plant.png")

    # 3. Healthy vs Diseased
    plt.figure(figsize=(8, 6))
    status_counts = df.groupby("Status")["Count"].sum()
    status_counts.plot(kind="pie", autopct='%1.1f%%', colors=["tomato", "lightgreen"], startangle=140)
    plt.title("Healthy vs Diseased Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("healthy_vs_diseased.png")
    print("Saved healthy_vs_diseased.png")

    # 4. Summary Table
    print("\nData Summary:")
    print(df.groupby("Plant")["Count"].sum().sort_values(ascending=False))
    print("\nTotal Images:", df["Count"].sum())

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        print(f"Error: Directory {BASE_DIR} not found.")
    else:
        df = collect_data(BASE_DIR)
        plot_data(df)
