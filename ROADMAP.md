# Football Edge Lab: Project Roadmap

This roadmap outlines the long-term vision for the Football Edge Lab, transitioning from a backtesting tool into a comprehensive betting strategy suite.

---

## Phase 1: Model Transparency & UI Control
*Focus on making the "Random Forest" predictable and configurable.*

### 1. Feature Importance Analytics
- **Backend**: Extract `feature_importances_` post-training.
- **UI**: Visualize the top 10-20 driving factors in a bar chart within the Training Panel.
- **Goal**: Allow users to see if the model is over-relying on specific features (e.g., just odds).

### 2. Hyperparameter Tuning
- **UI**: Add advanced toggles for `n_estimators`, `max_depth`, and `min_samples_leaf`.
- **Backend**: Pass these parameters from the API to the `RandomForestClassifier`.
- **Goal**: Enable direct experimentation with model complexity from the browser.

---

## Phase 2: Market Expansion & Data Fidelity
*Moving beyond the 1X2 market and refining data handling.*

### 3. Over/Under 2.5 & BTTS Markets
- **Engineering**: Add targets for total goals and both teams to score.
- **Modeling**: Multi-target training support (train and save separate models for each market).
- **UI**: Add a market selector to the Backtest and Live Prediction panels.

### 4. Automated Odds Fetching
- **Integration**: Connect to a sports odds API (e.g., The-Odds-API).
- **Workflow**: Fetch today’s fixtures and current market odds automatically.
- **Goal**: Eliminate manual data entry for live predictions.

---

## Phase 3: Live Tracking & Automation
*Bridging the gap between lab research and actual use.*

### 5. Paper Trading Tracker (The "Journal")
- **Feature**: "Save Prediction" button to log predictions into a local SQLite/JSON store.
- **Dashboard**: A performance tracker page showing actual ROI vs. Backtest ROI.
- **Workflow**: "Close Position" workflow to enter final scores and see P&L.

### 6. Performance Optimizations
- **Backend**: Implement caching for feature loading (using Arrow or Parquet for faster I/O).
- **Execution**: Parallelize feature engineering for large-scale historical datasets.

---

## Future Vision
- **Portfolio Management**: Kelly Criterion across multiple concurrent matches.
- **Advanced Models**: Experimenting with XGBoost or LightGBM for non-linear patterns.
- **Dockerization**: One-click deployment for local or cloud hosting.
