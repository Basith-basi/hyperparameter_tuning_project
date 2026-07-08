from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.tuning import objective
from src.train_final_model import train_best_model
from src.evaluation import evaluate_model

from src.utils import (
    save_best_parameters,
    save_model,
    save_study,
    save_optimization_history,
    save_parameter_importance,
    create_directories
)

import optuna


def main():

    # ==========================================
    # Create Required Folders
    # ==========================================
    create_directories()

    # ==========================================
    # Dataset Configuration
    # ==========================================
    DATA_PATH = "data/cancer_dataset.csv"      # Your dataset
    TARGET_COLUMN = "target"                   # Change if different

    # ==========================================
    # Load Dataset
    # ==========================================
    df = load_data(DATA_PATH)

    # ==========================================
    # Preprocess Dataset
    # ==========================================
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        target_column=TARGET_COLUMN
    )

    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")

    # ==========================================
    # Create Optuna Study
    # ==========================================
    study = optuna.create_study(
        direction="maximize",
         pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5
    )
    )

    print("\nStarting Hyperparameter Tuning...\n")

    # ==========================================
    # Run Hyperparameter Search
    # ==========================================
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=50,
        show_progress_bar=True
    )

    # ==========================================
    # Display Best Results
    # ==========================================
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 60)

    print(f"\nBest Cross Validation Accuracy : {study.best_value:.4f}")

    print("\nBest Hyperparameters:\n")

    for key, value in study.best_params.items():
        print(f"{key:<20}: {value}")

    # ==========================================
    # Save Best Parameters
    # ==========================================
    save_best_parameters(study.best_params)

    # ==========================================
    # Save Optuna Study
    # ==========================================
    save_study(study)

    # ==========================================
    # Train Final Model
    # ==========================================
    print("\nTraining Final Model...\n")

    model = train_best_model(
        X_train,
        y_train,
        study.best_params
    )

    # ==========================================
    # Save Final Model
    # ==========================================
    save_model(model)

    # ==========================================
    # Evaluate Model
    # ==========================================
    print("\nEvaluating Model...\n")

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # ==========================================
    # Save Optuna Plots
    # ==========================================
    save_optimization_history(study)

    save_parameter_importance(study)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()