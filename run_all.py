"""
run_all.py
==========
Convenience runner that executes all five pipeline stages in order.
Each stage is independent and can also be run on its own
(python3 stageN_*.py) as long as the stage before it has already run.
"""

import stage1_load_clean
import stage2_feature_engineering
import stage3_train
import stage4_evaluate
import stage5_interpret


def run():
    print("=" * 70 + "\nSTAGE 1: LOAD & CLEAN\n" + "=" * 70)
    stage1_load_clean.run()

    print("\n" + "=" * 70 + "\nSTAGE 2: FEATURE ENGINEERING\n" + "=" * 70)
    stage2_feature_engineering.run()

    print("\n" + "=" * 70 + "\nSTAGE 3: TRAIN\n" + "=" * 70)
    stage3_train.run()

    print("\n" + "=" * 70 + "\nSTAGE 4: EVALUATE\n" + "=" * 70)
    stage4_evaluate.run()

    print("\n" + "=" * 70 + "\nSTAGE 5: INTERPRET\n" + "=" * 70)
    stage5_interpret.run()

    print("\nPipeline complete. All outputs saved.")


if __name__ == "__main__":
    run()
