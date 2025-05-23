## Testing and Refinement Plan for Contrarian Trading Agent System

This document outlines the strategy for testing the newly developed and modified agents within the FinRobot framework, focusing on their contrarian trading capabilities. It includes unit test case outlines, an integration test scenario, and a workflow for backtesting and refinement.

### 1. Key Unit Test Cases

Unit tests will primarily use Python's `unittest` framework with `unittest.mock` to isolate components and simulate external dependencies (LLM calls, `akshare` API calls).

**a. `NewsAnalysisAgent.analyze_news_contrarian`**

*   **Test Case 1: Successful JSON Output with Stock Correlation and `end_date_override`**
    *   **Description:** Verify correct data fetching (mocked) up to `end_date_override`, prompt construction, and parsing of valid JSON from a (mocked) LLM response when news and stock symbols are provided.
    *   **Mock Setup:**
        *   `cn_news_utils.search_news` (or `get_financial_news`): Mock to return a pandas DataFrame with news items, ensuring timestamps allow filtering by `end_date_override`.
        *   `akshare_utils.get_stock_info`: Mock to return basic stock info.
        *   `akshare_utils.get_stock_history`: Mock to return a DataFrame with price history, checking that it's called with `end_date` derived from `end_date_override`.
        *   Mock `self.user_proxy.initiate_chat` to return a string representing the expected JSON structure (as defined in the agent's system message).
    *   **Assertions:**
        *   Verify `cn_news_utils` functions are (conceptually) providing news up to `end_date_override`.
        *   Verify `akshare_utils.get_stock_history` is called with an end date matching `end_date_override`.
        *   Verify the prompt to the LLM includes news summaries and price action summaries reflecting data up to `end_date_override`.
        *   Verify the output is a dictionary conforming to the agent's specified JSON structure.
        *   Ensure the `price_behavior_correlation` field correctly reflects the mock price data relative to the news and `end_date_override`.

*   **Test Case 2: Handling API Failures and Malformed LLM JSON Output**
    *   **Description:** Test robustness when external data fetching fails (e.g., for stock prices) and when the LLM returns a non-JSON or incorrectly structured JSON response.
    *   **Mock Setup:**
        *   `cn_news_utils.search_news` returns sample news.
        *   `akshare_utils.get_stock_history` raises an exception or returns an empty DataFrame.
        *   Mock the LLM call (`self.user_proxy.initiate_chat`) on different runs to return:
            1.  A plain text string (not JSON).
            2.  A malformed JSON string.
    *   **Assertions:**
        *   When stock history fails, verify `price_behavior_correlation` in the prompt indicates missing/failed stock data.
        *   When LLM output is a plain string, verify the method returns this raw string and a warning is (conceptually) logged.
        *   When LLM output is malformed JSON, verify the method returns the raw string and a warning is logged.

**b. `DeepValueContrarianPersonaAgent.analyze_opportunity`**

*   **Test Case 1: Strong BUY Recommendation on Contrarian Signal**
    *   **Description:** Ensure the agent outputs a 'BUY' recommendation with high confidence when provided with good fundamentals and strong contrarian signals (e.g., news analysis indicating "诱空" or fear-mongering).
    *   **Mock Setup:**
        *   `company_info`: Mock basic info.
        *   `financials_summary`: Mock strong fundamentals (e.g., low P/E, healthy D/E).
        *   `contrarian_sentiment_analysis`: Mock JSON output from `NewsAnalysisAgent` indicating high `manipulation_likelihood_score` and a `core_contrarian_signal` like "buy on fear".
        *   `market_forecast` (optional): Mock as neutral.
        *   Mock LLM (`self.deep_value_analyst`'s chat) to return a JSON string matching the agent's output structure, with `recommendation: "BUY"` and high `confidence_score`.
    *   **Assertions:**
        *   Verify the prompt sent to the LLM correctly includes all input data, especially the contrarian signals.
        *   Verify the returned dictionary has `recommendation: "BUY"`.
        *   Verify `analysis_summary.contrarian_opportunity_assessment` reflects the input contrarian signals.

*   **Test Case 2: HOLD Recommendation with Ambiguous Signals**
    *   **Description:** Test if the agent recommends 'HOLD' when fundamentals are decent but contrarian signals are weak or market forecast is negative.
    *   **Mock Setup:**
        *   `financials_summary`: Mock decent fundamentals.
        *   `contrarian_sentiment_analysis`: Mock low manipulation score, neutral signals.
        *   `market_forecast`: Mock as bearish.
        *   Mock LLM to return JSON indicating 'HOLD'.
    *   **Assertions:**
        *   Verify output `recommendation: "HOLD"`.
        *   Verify `detailed_reasoning` mentions the mixed/negative overriding signals.

**c. `MarketManipulationAnalystPersonaAgent.assess_manipulation`**

*   **Test Case 1: Identifies "诱多" (Pump and Dump)**
    *   **Description:** Verify the agent correctly identifies a "诱多" tactic (e.g., hyping a stock to offload at higher prices).
    *   **Mock Setup:**
        *   `contrarian_sentiment_analysis`: Mock JSON where `surface_sentiment` is overly positive for minor news, `price_behavior_correlation` shows price spikes on news but then fading, and `manipulation_likelihood_score` is high.
        *   `recent_price_volume_summary` (optional): Mock string noting unusual volume spikes on positive news followed by declines.
        *   Mock LLM (`self.manipulation_spotter`'s chat) to return JSON identifying "诱多".
    *   **Assertions:**
        *   Verify output JSON `suspected_manipulation_tactic` contains "诱多".
        *   Verify `key_indicators` list relevant factors.
        *   Verify `suggested_contrarian_action` suggests caution or selling.

*   **Test Case 2: No Significant Manipulation Detected**
    *   **Description:** Ensure the agent indicates no clear manipulation if signals are absent.
    *   **Mock Setup:**
        *   `contrarian_sentiment_analysis`: Mock with low manipulation score.
        *   Mock LLM to return JSON indicating low manipulation likelihood.
    *   **Assertions:**
        *   Verify `suspected_manipulation_tactic` suggests no clear manipulation.
        *   Verify `confidence_score` for manipulation is low.

**d. `RiskManagerAgent.assess_trade_risk`**

*   **Test Case 1: High Risk Output for Risky Trade**
    *   **Description:** Test if a BUY trade for a volatile stock that significantly increases portfolio concentration is flagged as high risk.
    *   **Mock Setup:**
        *   `proposed_trade`: BUY action, large quantity.
        *   `current_portfolio_summary`: Mock data showing the trade would lead to high concentration.
        *   Mock `self._calculate_volatility` to return a high value (e.g., 0.7).
        *   Mock `self._check_concentration` to return values indicating high concentration post-trade.
        *   Mock LLM (`self.risk_assessment_analyst`'s chat) to return a string including "Risk Level: High".
    *   **Assertions:**
        *   Verify `_calculate_volatility` and `_check_concentration` are called.
        *   Verify the prompt for the LLM includes the high volatility and concentration figures.
        *   Verify the final output string contains "Risk Level: High".

*   **Test Case 2: `_calculate_volatility` Correctness**
    *   **Description:** Test the internal volatility calculation.
    *   **Mock Setup:** Mock `akshare_utils.get_stock_history` to return a DataFrame with a known series of closing prices.
    *   **Assertions:** Verify the calculated annualized volatility is correct. Test with insufficient data (should return `None`).

*   **Test Case 3: `_check_concentration` Correctness**
    *   **Description:** Test internal concentration calculations for stock and sector.
    *   **Mock Setup:** Provide `symbol`, `proposed_trade_value`, and `portfolio_summary` with various existing holdings. Mock `akshare_utils.get_stock_info` for industry lookups.
    *   **Assertions:** Verify `stock_concentration` and `sector_concentration` are calculated correctly for new buys and additions to existing holdings.

**e. `PortfolioManagerAgent.generate_trade_decision_contrarian`**

*   **Test Case 1: Contrarian BUY Decision, Adjusted by Risk**
    *   **Description:** End-to-end logic: strong contrarian signals lead to an initial BUY, which is then quantity-adjusted due to a "Medium" risk assessment.
    *   **Mock Setup:**
        *   Mock all input analysis results (market, technical, contrarian news, personas) to strongly suggest a "诱空" situation and good underlying value.
        *   Mock the Head Trader LLM call to return an initial BUY JSON with a specific quantity (e.g., 100 shares).
        *   Mock `self.risk_manager_agent.assess_trade_risk` to return a string indicating "Risk Level: Medium".
    *   **Assertions:**
        *   Verify the prompt to the Head Trader LLM correctly summarizes the contrarian inputs.
        *   Verify the `initial_recommendation` part of the output is the BUY JSON.
        *   Verify `risk_assessment.summary` contains the "Risk Level: Medium" text.
        *   Verify `final_decision_after_risk.proposed_trade_details.quantity_suggestion` is reduced (e.g., to "75 shares").
        *   Verify the overall output is a correctly structured JSON.

*   **Test Case 2: HOLD Decision Propagated**
    *   **Description:** If initial analyses are mixed and Head Trader decides HOLD, this should be the final outcome.
    *   **Mock Setup:** Mock inputs to be inconclusive. Mock Head Trader LLM to return a HOLD decision JSON.
    *   **Assertions:**
        *   `risk_manager_agent.assess_trade_risk` should NOT be called.
        *   The `final_decision_after_risk.decision` should be "HOLD".
        *   `final_decision_after_risk.proposed_trade_details` should be `None` or omitted.

**f. `Backtester._simulate_trade`**

*   **Test Case 1: Correct Portfolio Update on BUY**
    *   **Description:** Verify cash reduction and holding creation/update.
    *   **Mock Setup:** `portfolio` with initial cash. `trade_details` for BUY. `trade_price`.
    *   **Assertions:** Check `portfolio['cash']`, `portfolio['holdings'][symbol]['quantity']`, `portfolio['holdings'][symbol]['average_cost']`, and `trade_log` entry.

*   **Test Case 2: Correct Portfolio Update on SELL (Full and Partial)**
    *   **Description:** Verify cash increase and holding reduction/removal.
    *   **Mock Setup:** `portfolio` with existing holdings. `trade_details` for SELL. `trade_price`.
    *   **Assertions:** Check portfolio and `trade_log` after a partial sell, then after selling the remainder.

**g. `Backtester.calculate_performance_metrics`**

*   **Test Case 1: Basic Metrics Calculation**
    *   **Description:** Verify calculation of final value, total return %, and number of trades.
    *   **Mock Setup:** Set `initial_capital`, populate `daily_portfolio_value` list, and `trade_log`.
    *   **Assertions:** Check calculated values in the returned metrics dictionary.

### 2. Integration Test Scenario

*   **Stock Symbol & Date:**
    *   Symbol: `000XYZ` (Fictional "Acme Innovations")
    *   Hypothetical Analysis Date: `2023-07-15` (for point-in-time data fetching simulation)

*   **Mock Inputs & Expected Flow:**

    1.  **Data Fetching (Simulated for `2023-07-15`):**
        *   **News (`cn_news_utils.search_news` & `get_financial_news` via `NewsAnalysisAgent`):**
            *   Mock `cn_news_utils` to return 2-3 news items for "Acme Innovations" or "000XYZ" with timestamps up to `2023-07-15`.
                *   News 1 (Published `2023-07-14`): "Acme Innovations (000XYZ) stock plunges 10% on rumors of failed clinical trial. Analysts express deep concerns." (Surface: Very Negative)
                *   News 2 (Published `2023-07-15`): "Sources inside Acme Innovations anonymously claim trial data was misinterpreted; official statement pending." (Surface: Neutral/Slightly Positive, but source is weak)
        *   **Price History (`akshare_utils.get_stock_history` via `NewsAnalysisAgent`):**
            *   Mock data for `000XYZ` ending `2023-07-15`:
                *   `2023-07-13`: Close 50.00
                *   `2023-07-14`: Close 45.00 (Plunges 10%)
                *   `2023-07-15`: Close 43.00 (Further slight drop, high volume)

    2.  **`NewsAnalysisAgent.analyze_news_contrarian` (for `000XYZ`, `end_date_override='2023-07-15'`):**
        *   **Mock LLM Output:**
            ```json
            {
              "news_summary": "Stock plunged on trial rumors, but an anonymous source suggests misinterpretation. Price continued slight decline.",
              "surface_sentiment": {"type": "负面", "score": 2},
              "price_behavior_correlation": "股价在负面新闻后暴跌，匿名澄清后仍小幅下跌，但成交量放大。",
              "manipulation_analysis": {
                "discrepancy_observation": "主要负面新闻导致暴跌，但匿名澄清后股价未显著反弹，反而小幅续跌伴随高成交量，可能表明市场仍在消化或存在持续抛压/吸筹。",
                "exaggeration_煽动性_fud_factor": "初次暴跌新闻可能被市场过度解读或利用以制造恐慌。",
                "dominant_force_hypothesis": "可能存在利用坏消息进行诱空吸筹，或恐慌盘持续涌出。",
                "reasoning": "匿名消息的出现和股价在高成交量下的小幅续跌，而非反弹，值得关注。"
              },
              "manipulation_likelihood_score": 7,
              "true_impact_assessment": "若匿名消息属实且官方能澄清，当前股价可能被低估。短期内不确定性高。",
              "core_contrarian_signal": "警惕恐慌性抛售，若基本面未变且能澄清，可能是潜在的错杀机会。",
              "supporting_data_points": ["Stock plunges 10% on rumors", "Sources inside Acme...claim trial data was misinterpreted", "成交量放大"]
            }
            ```
        *   **Checkpoint:** Verify the agent's output JSON contains high `manipulation_likelihood_score` and a `core_contrarian_signal` pointing to potential mispricing.

    3.  **Other Agents (mocked outputs for `2023-07-15` for `000XYZ`):**
        *   **`TechnicalAnalysisAgent`:** "Stock is heavily oversold (RSI < 20). MACD shows deep bearish momentum but potential for divergence if price stabilizes. Multiple support levels broken. Extreme caution advised, but watch for signs of capitulation."
        *   **`MarketForecasterAgent`:** "Highly Uncertain. Sector outlook stable, but company-specific news is driving volatility. Prediction: Volatile with downward bias unless positive clarification emerges. Range: 40-48 for next week."
        *   **Financial Summary (Mock for `DeepValueContrarianPersonaAgent`):** `P/E: 8` (previously 15 before drop), `P/B: 0.9`, `Debt/Equity: 0.2`. `notes`: "Strong pipeline excluding this trial, solid cash reserves."
        *   **`DeepValueContrarianPersonaAgent.analyze_opportunity` (fed with above):**
            *   **Mock LLM Output:** Recommendation: "BUY", Confidence: 0.75. Reasoning: "Fundamentals appear solid and undervalued after price drop. Contrarian news analysis suggests potential fear-mongering. If trial issues are clarified, upside is significant. This is a calculated risk on information asymmetry."
            *   **Checkpoint:** Verify its reasoning focuses on undervaluation due to panic and the contrarian signal.
        *   **`MarketManipulationAnalystPersonaAgent.assess_manipulation` (fed with `NewsAnalysisAgent` output):**
            *   **Mock LLM Output:** `suspected_manipulation_tactic`: "利用负面（可能被夸大的）消息诱空，并可能伴随低位吸筹迹象 (高成交量下未持续暴跌)", `confidence_score`: 0.8. `suggested_contrarian_action`: "寻找企稳迹象，考虑左侧交易或等待澄清后介入".
            *   **Checkpoint:** Verify it identifies "诱空" and suggests a contrarian positive action.

    4.  **`PortfolioManagerAgent.generate_trade_decision_contrarian`:**
        *   **Inputs:** All the above agent outputs. `current_portfolio_summary` shows sufficient cash.
        *   **Checkpoint (Prompt for Head Trader):** Verify the prompt correctly aggregates all the above analyses, especially the strong contrarian signals from news and persona agents.
        *   **Mock LLM (Head Trader - Initial Decision):**
            ```json
            {
              "symbol_analyzed": "000XYZ",
              "decision_pre_risk": "BUY",
              "confidence_pre_risk": 0.7,
              "reasoning_pre_risk": "Strong contrarian signals from News Analysis (manipulation score 7) and Persona Agents (Deep Value BUY, Manipulation Analyst suggests '诱空'). Technicals show oversold. Financials appear resilient post-drop. Betting on mispricing due to panic/manipulation.",
              "proposed_trade_details_pre_risk": {"action": "BUY", "symbol": "000XYZ", "quantity_suggestion": "Allocate 3% of portfolio", "price_target": 55.0, "stop_loss": 40.0}
            }
            ```
        *   **Checkpoint:** Head Trader's initial decision is BUY, reasoning emphasizes contrarian elements.

    5.  **`RiskManagerAgent.assess_trade_risk` (if initial decision is BUY/SELL):**
        *   **Inputs:** The `proposed_trade_details_pre_risk`. Assume volatility is calculated as high (0.6) due to recent plunge. Concentration is low.
        *   **Mock LLM Output:** "Risk Level: High. Rationale: Extreme recent volatility and negative market perception due to trial rumors. While there are contrarian signals, the uncertainty remains very high until official clarification. Potential for further downside if rumors are confirmed."
        *   **Checkpoint:** Risk is assessed as High.

    6.  **`PortfolioManagerAgent` (Final Output):**
        *   **Checkpoint:**
            *   `initial_recommendation` matches the Head Trader's output.
            *   `risk_assessment.risk_level` is "High".
            *   `final_decision_after_risk.decision` is still "BUY" (assuming Head Trader's conviction is high for contrarian plays), BUT `final_decision_after_risk.proposed_trade_details.quantity_suggestion` is reduced (e.g., "Allocate 1.5% of portfolio" - halved due to High risk).
            *   `final_decision_after_risk.reasoning` includes the risk manager's concerns and the rationale for quantity reduction.

### 3. Backtesting & Refinement Workflow

1.  **Setup a Backtest Run:**
    *   **Parameters:**
        *   `llm_config`: Use a consistent (potentially less costly for multiple runs) LLM configuration.
        *   `start_date_str`, `end_date_str`: Select a period of at least 6-12 months, covering different market conditions (bullish, bearish, sideways).
        *   `initial_capital`: A standard figure, e.g., $100,000.
        *   `stock_symbols`: A diverse list of 5-10 stocks across various sectors, including some known for being news-sensitive or having historical instances of strong sentiment-driven moves.
    *   **Data:** Ensure historical price data (`akshare_utils.get_stock_history`) and news data (`cn_news_utils` via `NewsAnalysisAgent`) can be reliably fetched or mocked for the chosen period and symbols. The `end_date_override` in agents is key here.

2.  **Execution and Initial Review:**
    *   Run the `Backtester.run_backtest()`.
    *   **Trade Log Analysis (`trade_log`):**
        *   Were trades executed? For which stocks and on what dates?
        *   Crucially, for each trade, examine the *full JSON output of `PortfolioManagerAgent.generate_trade_decision_contrarian`* (this might require temporarily logging it in the backtester). This output contains the chain of reasoning, including the initial decision, risk assessment, and final decision with quantity.
        *   Are the reasons for trades aligning with the intended contrarian strategy? (e.g., buying on high manipulation scores, negative surface sentiment but positive deep value assessment).
    *   **Performance Metrics (`calculate_performance_metrics`):**
        *   Overall P&L, Total Return %.
        *   Number of trades.
        *   (If implemented later) Sharpe ratio, max drawdown, win/loss ratio.
        *   Compare against a benchmark (e.g., buy-and-hold the selected stocks or a relevant market index like CSI 300).

3.  **Systematic Troubleshooting and Refinement (Iterative Loop):**

    *   **Identify Anomalies/Suboptimal Behavior:**
        *   *Missed Opportunities:* The system identified strong "诱空" signals (e.g., via `NewsAnalysisAgent` and `MarketManipulationAnalystPersonaAgent`) but `PortfolioManagerAgent` decided "HOLD" or "WAIT".
        *   *Bad Trades:* The system bought into a "诱多" trap, resulting in a loss when the price subsequently fell.
        *   *Risk Mismanagement:* The system took on too large a position in a high-risk situation, or was too timid when risk was assessed as low/medium for a good opportunity.
        *   *Inconsistent Behavior:* Similar scenarios leading to different outcomes.

    *   **Deep Dive into a Specific Decision Point (Example: Missed "诱空" BUY):**
        1.  **Isolate the Day/Stock:** From the backtest logs (or by re-running with extensive logging for that day), get all agent inputs and outputs for that specific decision.
        2.  **`PortfolioManagerAgent` Output:** Examine `initial_recommendation` and `final_decision_after_risk`. Why did it not BUY? Was `reasoning_pre_risk` unconvincing? Was `confidence_pre_risk` too low? Did `RiskManagerAgent` override it or suggest extreme caution?
        3.  **Inputs to `PortfolioManagerAgent`:**
            *   Check `contrarian_sentiment_analysis` from `NewsAnalysisAgent`: Was `manipulation_likelihood_score` high enough? Was `core_contrarian_signal` clear?
            *   Check `persona_analyses`: Did `DeepValueContrarianPersonaAgent` recommend BUY? Did `MarketManipulationAnalystPersonaAgent` confirm "诱空"?
            *   Check `technical_analysis` and `market_forecast`: Were these overwhelmingly negative, causing the `PortfolioManagerAgent` (Head Trader) to be too cautious despite contrarian signals?
        4.  **Examine LLM Prompts:**
            *   The prompt fed to the `PortfolioManagerAgent` (Head Trader). Did it adequately emphasize the contrarian signals?
            *   The prompts fed to the Persona agents. Did they receive the correct information from upstream agents?
        5.  **Identify the Weak Link:**
            *   *News Analysis:* Maybe `NewsAnalysisAgent`'s prompt needs to be more sensitive to detect subtle manipulation clues, or its JSON output needs to be clearer for downstream consumption.
            *   *Persona Agents:* Perhaps the `DeepValueContrarianPersonaAgent` needs its system prompt adjusted to be more aggressive on strong manipulation signals if fundamentals are at least "fair". Or `MarketManipulationAnalystPersonaAgent` needs to provide a more actionable `suggested_contrarian_action`.
            *   *Portfolio Manager (Head Trader):* This is often the key. Its system prompt might need adjustment to give more explicit weight to contrarian factors when they conflict with traditional indicators. For example: "If `manipulation_likelihood_score` from NewsAnalysis is > X AND DeepValueContrarian recommends BUY, you should strongly consider a BUY unless technicals/market forecast are disastrous. Your reasoning must explicitly state how contrarian signals influenced the decision, especially if overriding other indicators."
            *   *Risk Manager:* Was its assessment appropriate? If it was "High" risk leading to no trade, was that correct, or is it too sensitive for this contrarian strategy? Its heuristic impact on quantity might need tuning.

    *   **Propose and Implement Changes:**
        *   Modify the system message of the identified agent(s).
        *   Adjust heuristic rules in `PortfolioManagerAgent` for how it reacts to risk levels.
        *   Change how data is summarized or presented in prompts.

    *   **Re-run and Compare:**
        *   Run the backtest again with the same parameters and period.
        *   Compare the specific trade decision that was problematic. Did it change as expected?
        *   Compare overall metrics. Did the change improve performance, or did it have unintended negative consequences elsewhere?
        *   Consider running on an out-of-sample period if the change seems good on the initial test period to check for overfitting the refinement to that specific period.

    *   **Document:** Keep a log of all refinements, the rationale, and their impact on performance. This is crucial for understanding the system's evolution and avoiding cycles.

This systematic approach of diagnosis, targeted refinement, and iterative testing is key to improving the complex multi-agent system.
