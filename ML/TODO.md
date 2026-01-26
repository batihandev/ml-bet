# TODO (ML)

- Add permutation importance (more reliable).
- Run feature ablation (remove top 1–3 features and compare OOF/Brier/ROI).

### Strategy: Dynamic Block of 4 (MBC 3 Optimized)

**1. Configuration**

- **Cron Schedule:** Daily @ 00:00.
- **Lookahead Window:** 72 Hours (3 Days).
- **Bankroll Mgmt:** 1 Block = 5 Units (1 Unit per Ticket).
- **Max Exposure:** A single Match ID can appear in max **2 active blocks**.

**2. Execution Logic (The Script)**

- **Fetch:** Get all fixtures for `[Now, Now + 72h]`.
- **Filter:** Remove matches with `Risk > Threshold` or `Odds < 1.30`.
- **Rank:** Sort remaining matches by `Signal_Strength` (descending).
- **Select:** Take top 4 Matches.
  - _Check:_ If any match exceeds `Max Exposure`, skip and take next.
- **Action:** If 4 matches selected:
  - **Ticket 1:** A + B + C
  - **Ticket 2:** A + B + D
  - **Ticket 3:** A + C + D
  - **Ticket 4:** B + C + D
  - **Ticket 5:** A + B + C + D
- **Log:** Save Match IDs to `active_bets.db` to track exposure.

**3. Success Metrics**

- **3 Wins:** Breakeven / Small Profit (The Safety Net).
- **4 Wins:** Maximum Profit (The Goal).
