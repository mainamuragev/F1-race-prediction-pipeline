"""
main.py
-------
Pipeline orchestrator. Runs every step in the correct order:

  1. clean_data      — fix Ergast nulls, coerce types
  2. build_features  — join tables, compute all rolling features
  3. train_model     — time-based split, train GBR, save model + scaler
  4. build_upcoming  — build per-driver feature rows for next race
  5. predict         — score upcoming features, print leaderboard

Run the full pipeline:
    python main.py

Run individual steps:
    python main.py --step clean
    python main.py --step features
    python main.py --step train
    python main.py --step upcoming
    python main.py --step predict
"""

import argparse
import sys
import time


def step_clean():
    print("\n" + "="*60)
    print("STEP 1: Clean raw data")
    print("="*60)
    from src.processing.clean_data import run_all
    run_all()


def step_features():
    print("\n" + "="*60)
    print("STEP 2: Build unified feature set")
    print("="*60)
    from src.features.build_features import build_features
    build_features()


def step_train():
    print("\n" + "="*60)
    print("STEP 3: Train model")
    print("="*60)
    from src.models.train_model import train_model
    train_model()


def step_upcoming():
    print("\n" + "="*60)
    print("STEP 4: Build upcoming race features")
    print("="*60)
    from src.features.build_upcoming_features import build_upcoming_features
    build_upcoming_features()


def step_predict():
    print("\n" + "="*60)
    print("STEP 5: Predict upcoming race")
    print("="*60)
    from src.models.predict_upcoming import predict_upcoming
    predict_upcoming()


STEPS = {
    "clean":    step_clean,
    "features": step_features,
    "train":    step_train,
    "upcoming": step_upcoming,
    "predict":  step_predict,
}


def run_full_pipeline():
    start = time.time()
    for name, fn in STEPS.items():
        try:
            fn()
        except Exception as e:
            print(f"\n❌ Step '{name}' failed: {e}")
            print("   Fix the error above and re-run, or run individual steps with --step")
            sys.exit(1)
    elapsed = time.time() - start
    print(f"\n🏁 Full pipeline complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="F1 Race Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Run a single pipeline step instead of the full pipeline",
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
