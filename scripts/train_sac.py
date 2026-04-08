"""
train_sac.py — SAC baseline training for OrbitalInsertionEnv.

Trains a Soft Actor-Critic agent on planetary orbital insertion.
Supports three experimental conditions:

  fixed     : single planet (Earth by default), fastest convergence
  random    : random planets, no curriculum
  curriculum: random planets ordered easy → hard by habitability score

Usage
-----
    # Quick test (1k steps, ~30 seconds)
    python scripts/train_sac.py --mode fixed --steps 1000 --tag test

    # Earth baseline (recommended first run, ~30 min)
    python scripts/train_sac.py --mode fixed --steps 200000 --tag earth_baseline

    # Full curriculum run (overnight)
    python scripts/train_sac.py --mode curriculum --steps 500000 --tag curriculum_v1

    # Random planets (generalisation baseline)
    python scripts/train_sac.py --mode random --steps 500000 --tag random_v1

Outputs (saved to ./training_runs/<tag>/)
------------------------------------------
    config.json         — full experiment config + env fingerprints
    learning_curve.csv  — step, ep_reward, success_rate, fuel_eff, hab_bonus
    learning_curve.png  — publication-ready training figure
    model_final.zip     — trained SB3 SAC model
    model_best.zip      — best checkpoint (by eval reward)
    eval_results.json   — final evaluation over 50 episodes
"""

from exorl.commands.train_sac import main


if __name__ == "__main__":
    main()
