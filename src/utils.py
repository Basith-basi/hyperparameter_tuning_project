import os
import json
import joblib
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as optuna_plot


# ============================================
# Create Required Directories
# ============================================

def create_directories():
    """
    Create output folders if they don't exist.
    """

    folders = [
        "outputs",
        "plots",
        "models"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# ============================================
# Save Best Hyperparameters
# ============================================

def save_best_parameters(best_params):
    """
    Save Optuna best parameters to JSON file.
    """

    create_directories()

    with open("outputs/best_parameters.json", "w") as file:
        json.dump(best_params, file, indent=4)

    print(" Best parameters saved to outputs/best_parameters.json")


# ============================================
# Save Trained Model
# ============================================

def save_model(model):
    """
    Save trained model.
    """

    create_directories()

    joblib.dump(model, "models/best_model.pkl")

    print(" Model saved to models/best_model.pkl")


# ============================================
# Save Optuna Study
# ============================================

def save_study(study):
    """
    Save Optuna study.
    """

    create_directories()

    joblib.dump(study, "outputs/study.pkl")

    print(" Study saved to outputs/study.pkl")


# ============================================
# Optimization History Plot
# ============================================

def save_optimization_history(study):
    """
    Save Optimization History Plot.
    """

    create_directories()

    plt.figure(figsize=(10, 6))

    optuna_plot.plot_optimization_history(study)

    plt.tight_layout()

    plt.savefig("plots/optimization_history.png")

    plt.close()

    print(" Optimization history saved.")


# ============================================
# Parameter Importance Plot
# ============================================

def save_parameter_importance(study):
    """
    Save parameter importance plot.
    """

    create_directories()

    try:
        plt.figure(figsize=(10, 6))

        optuna_plot.plot_param_importances(study)

        plt.tight_layout()

        plt.savefig("plots/parameter_importance.png")

        plt.close()

        print("Parameter importance plot saved.")

    except RuntimeError:
        print(
            "Parameter importance plot skipped "
            "(all trials have identical scores)."
        )

    except Exception as e:
        print(f"Unable to generate parameter importance plot: {e}")

    print("Parameter importance saved.")


# ============================================
# Save Confusion Matrix Plot
# ============================================

def save_confusion_matrix(cm):
    """
    Save confusion matrix image.
    """

    create_directories()

    plt.figure(figsize=(6, 6))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.colorbar()

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("plots/confusion_matrix.png")

    plt.close()

    print(" Confusion Matrix saved.")