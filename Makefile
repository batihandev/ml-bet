UV=uv
PROJECT_DIR=ML

# default target
help:
	@echo "Available make targets:"
	@echo " make install     - install python dependencies"
	@echo " make dataset     - run dataset build script"
	@echo " make features    - run feature generation script"
	@echo " make train       - run training script"
	@echo " make predict     - run prediction script"
	@echo " make backend     - run the api backend"
	@echo " make clean       - remove uv artifacts"

# install dependencies
install:
	$(UV) sync --project $(PROJECT_DIR)

dataset:
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/dataset/cli.py

features:
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/features/pipeline.py

train:
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/training/cli.py

train_last5:
	TRAIN_START_DATE=$(shell date -d '5 years ago' +%Y-%m-%d) \
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/train_model.py

predict:
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/prediction/cli.py

clean:
	rm -rf $(PROJECT_DIR)/.venv $(PROJECT_DIR)/uv.lock



# ------------------------------------------------------------------
# Examples:
#
# 1) Last ~3 seasons, stricter threshold (0.6), save CSV + MD:
#    make backtest START=2022-07-01 END=2025-06-01 THR=0.6 OUT=backtest_all9_2022_2025_0p6.csv
#
# 2) Full validation period with milder threshold (0.5), save CSV + MD:
#    make backtest START=2018-08-01 END=2025-06-01 THR=0.5 OUT=backtest_all9_2018_2025_0p5.csv
#
# 3) Only recent season, higher threshold (0.65), no files (everything to /dev/null):
#    make backtest START=2024-07-01 END=2025-06-01 THR=0.65 OUT=/dev/null
#    (CSV and MD are both written to /dev/null)
# ------------


# Backtest FT 1X2 over a date range using model_ft_* and odds.
# Usage: make backtest_ft START=YYYY-MM-DD END=YYYY-MM-DD EDGE=0.02 STAKE=1 OUT_CSV=backtest_ft_1x2_bets.csv OUT_MD=backtest_ft_1x2_summary.md
# Example1 (flat, conservative): make backtest_ft START=2018-08-01 END=2025-06-01 EDGE=0.03 STAKE=1 OUT_CSV=ft_1x2_2018_2025_e003_bets.csv OUT_MD=ft_1x2_2018_2025_e003.md
# Example2 (flat, realistic edge): make backtest_ft START=2020-07-01 END=2025-06-01 EDGE=0.02 STAKE=1 OUT_CSV=ft_1x2_2020_2025_e002_bets.csv OUT_MD=ft_1x2_2020_2025_e002.md
# Example3 (flat, high edge filter): make backtest_ft START=2022-07-01 END=2025-06-01 EDGE=0.05 STAKE=1 OUT_CSV=ft_1x2_2022_2025_e005_bets.csv OUT_MD=ft_1x2_2022_2025_e005.md
# Example4 (quarter Kelly on high-edge only): make backtest_ft START=2020-07-01 END=2025-06-01 EDGE=0.03 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2020_2025_e003_k025_bets.csv OUT_MD=ft_1x2_2020_2025_e003_k025.md
# Example5 (half Kelly, research only): make backtest_ft START=2020-07-01 END=2025-06-01 EDGE=0.04 STAKE=1 KELLY_MULT=0.50 OUT_CSV=ft_1x2_2020_2025_e004_k050_bets.csv OUT_MD=ft_1x2_2020_2025_e004_k050.md
# Example6 (no files, console only): make backtest_ft START=2024-07-01 END=2025-06-01 EDGE=0.03 STAKE=1 OUT_CSV=/dev/null OUT_MD=/dev/null

# Low edge threshold: most inclusive, many small edges. 
# Expect many bets, low volatility, small ROI per bet.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.015 STAKE=1 OUT_CSV=ft_1x2_2023_2025_e0015_bets.csv OUT_MD=ft_1x2_2023_2025_e0015.md
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.015 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2023_2025_e0015_bets.csv OUT_MD=ft_1x2_2023_2025_e0015.md
# Moderate edge: good baseline test. Filters weak edges while preserving volume.
# Expect fewer bets, higher EV per bet, moderate variance.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.025 STAKE=1 OUT_CSV=ft_1x2_2023_2025_e0025_bets.csv OUT_MD=ft_1x2_2023_2025_e0025.md

# Very high edge: strong confidence threshold. 
# Expect few bets but best theoretical EV; variance increases.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.060 STAKE=1 OUT_CSV=ft_1x2_2023_2025_e0060_bets.csv OUT_MD=ft_1x2_2023_2025_e0060.md


# Quarter Kelly with mild edge: controlled staking using fractional Kelly.
# Expect conservative bankroll growth, drawdowns unlikely.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.020 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2023_2025_e0020_k025_bets.csv OUT_MD=ft_1x2_2023_2025_e0020_k025.md

# Quarter Kelly with higher edge cutoff: conservative but selective.
# Protects bankroll while targeting higher EV spots.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.035 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2023_2025_e0035_k025_bets.csv OUT_MD=ft_1x2_2023_2025_e0035_k025.md

# Half Kelly with strict edge: faster bankroll growth but higher drawdown risk.
# Recommended only if calibration strong & bankroll resilient.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.050 STAKE=1 KELLY_MULT=0.50 OUT_CSV=ft_1x2_2023_2025_e0050_k050_bets.csv OUT_MD=ft_1x2_2023_2025_e0050_k050.md

# Aggressive Kelly (0.9) with high edge threshold: speculative stress-test only.
# High risk/high reward; volatility can liquidate bankroll if model weak.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.060 STAKE=1 KELLY_MULT=0.90 OUT_CSV=ft_1x2_2023_2025_e0060_k090_bets.csv OUT_MD=ft_1x2_2023_2025_e0060_k090.md

# Smoke test: useful when developing to verify speed/backtest logic only.
# No files written; console summary only.
# make backtest_ft START=2023-01-01 END=2025-06-01 EDGE=0.030 STAKE=1 OUT_CSV=/dev/null OUT_MD=/dev/null

backtest_ft:
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/backtest/cli.py \
		--start-date $(START) \
		--end-date $(END) \
		--min-edge $(EDGE) \
		--stake $(STAKE) \
		$(if $(KELLY_MULT),--kelly-mult $(KELLY_MULT),) \
		--out-csv $(OUT_CSV) \
		--out-md $(OUT_MD)


# Train models on 5-year history up to START, with last ~5 months used for validation only.
# Usage: make train_window_5y START=2024-12-31
train_window_5y:
	TRAIN_START_DATE=$$(date -d "$(START) -5 years" +%Y-%m-%d) \
	TRAIN_END_DATE=$(START) \
	FIXED_CUTOFF_DATE=$$(date -d "$(START) -4 months" +%Y-%m-%d) \
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/train_model.py


# Low edge threshold: most inclusive, many small edges. Expect many bets, low volatility, small ROI per bet.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.015 STAKE=1 OUT_CSV=ft_1x2_2024_2025_e0015_bets.csv OUT_MD=ft_1x2_2024_2025_e0015.md
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.015 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2024_2025_e0015_bets.csv OUT_MD=ft_1x2_2024_2025_e0015.md

# Moderate edge: good baseline test. Filters weak edges while preserving volume. Expect fewer bets, higher EV per bet, moderate variance.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.025 STAKE=1 OUT_CSV=ft_1x2_2024_2025_e0025_bets.csv OUT_MD=ft_1x2_2024_2025_e0025.md

# Very high edge: strong confidence threshold. Expect few bets but best theoretical EV; variance increases.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.060 STAKE=1 OUT_CSV=ft_1x2_2024_2025_e0060_bets.csv OUT_MD=ft_1x2_2024_2025_e0060.md

# Quarter Kelly with mild edge: controlled staking using fractional Kelly. Expect conservative bankroll growth, drawdowns unlikely.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.020 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2024_2025_e0020_k025_bets.csv OUT_MD=ft_1x2_2024_2025_e0020_k025.md

# Quarter Kelly with higher edge cutoff: conservative but selective. Protects bankroll while targeting higher EV spots.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.035 STAKE=1 KELLY_MULT=0.25 OUT_CSV=ft_1x2_2024_2025_e0035_k025_bets.csv OUT_MD=ft_1x2_2024_2025_e0035_k025.md

# Half Kelly with strict edge: faster bankroll growth but higher drawdown risk. Recommended only if calibration strong & bankroll resilient.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.050 STAKE=1 KELLY_MULT=0.50 OUT_CSV=ft_1x2_2024_2025_e0050_k050_bets.csv OUT_MD=ft_1x2_2024_2025_e0050_k050.md

# Aggressive Kelly (0.9) with high edge threshold: speculative stress-test only. High risk/high reward; volatility can liquidate bankroll if model weak.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.060 STAKE=1 KELLY_MULT=0.90 OUT_CSV=ft_1x2_2024_2025_e0060_k090_bets.csv OUT_MD=ft_1x2_2024_2025_e0060_k090.md

# Smoke test: useful when developing to verify speed/backtest logic only. No files written; console summary only.
# make backtest_ft START=2024-12-31 END=2025-06-01 EDGE=0.030 STAKE=1 OUT_CSV=/dev/null OUT_MD=/dev/null


# Train models on 2020-01-01 .. 2024-12-31, with last ~6 months as internal validation.
# This is the training setup for backtesting 2025-01-01 .. 2025-06-01.
train_for_2025H1:
	TRAIN_START_DATE=2020-01-01 \
	TRAIN_END_DATE=2024-06-01 \
	FIXED_CUTOFF_DATE=2023-12-01 \
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/train_model.py

train_for_long:
	TRAIN_START_DATE=2016-07-15 \
	TRAIN_END_DATE=2024-06-01 \
	FIXED_CUTOFF_DATE=2023-12-01 \
	$(UV) run --project $(PROJECT_DIR) python $(PROJECT_DIR)/src/training/cli.py


backend:
	PYTHONPATH=$(PROJECT_DIR)/src $(UV) run --project $(PROJECT_DIR) python -m uvicorn api.main:app --reload --port 8050