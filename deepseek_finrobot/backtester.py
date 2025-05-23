import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
import json # For potential JSON parsing/dumping if needed in helpers

from deepseek_finrobot.agents import (
    PortfolioManagerAgent,
    NewsAnalysisAgent,
    TechnicalAnalysisAgent,
    DeepValueContrarianPersonaAgent,
    MarketManipulationAnalystPersonaAgent,
    RiskManagerAgent
)
from deepseek_finrobot.data_source import akshare_utils # Assuming get_stock_history is here
from deepseek_finrobot.utils import get_llm_config_for_autogen # If needed for agent setup

class Backtester:
    """
    A basic framework for backtesting trading strategies using FinRobot agents.
    """

    def __init__(self, llm_config: Dict[str, Any], start_date_str: str, end_date_str: str, 
                 initial_capital: float, stock_symbols: List[str]):
        """
        Initializes the Backtester.

        Args:
            llm_config: Configuration for initializing LLM-based agents.
            start_date_str: Backtest start date in "YYYY-MM-DD" format.
            end_date_str: Backtest end date in "YYYY-MM-DD" format.
            initial_capital: Starting capital for the backtest.
            stock_symbols: A list of stock symbols to trade.
        """
        self.llm_config = llm_config
        self.start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        self.end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        self.initial_capital = initial_capital
        self.stock_symbols = stock_symbols

        # Initialize agents
        self.risk_manager_agent = RiskManagerAgent(llm_config=self.llm_config)
        self.portfolio_manager_agent = PortfolioManagerAgent(llm_config=self.llm_config, risk_manager_agent_instance=self.risk_manager_agent)
        self.news_analysis_agent = NewsAnalysisAgent(llm_config=self.llm_config)
        self.technical_analysis_agent = TechnicalAnalysisAgent(llm_config=self.llm_config)
        self.deep_value_agent = DeepValueContrarianPersonaAgent(llm_config=self.llm_config)
        self.manipulation_spotter_agent = MarketManipulationAnalystPersonaAgent(llm_config=self.llm_config)
        
        # Portfolio and logs
        self.portfolio: Dict[str, Any] = {
            "cash": self.initial_capital, 
            "holdings": {}, # Stores dicts like {"symbol": {"quantity": X, "average_cost": Y}}
            "total_value": self.initial_capital
        }
        self.trade_log: List[Dict[str, Any]] = []
        self.daily_portfolio_value: List[Dict[str, Any]] = []
        
        self.historical_price_data: Dict[str, pd.DataFrame] = {} # To store fetched price data

        print(f"Backtester initialized: {start_date_str} to {end_date_str} with ${initial_capital:,.2f} for symbols: {', '.join(stock_symbols)}")

    def _get_point_in_time_news_analysis(self, symbol: str, company_name: str, current_date: datetime.date) -> Dict:
        """
        Fetches contrarian news analysis for a symbol at a specific point in time.
        (Requires NewsAnalysisAgent to be modified for end_date_override)
        """
        print(f"[{current_date}] Getting news analysis for {symbol}...")
        try:
            # Assuming company_name can be derived or is passed (e.g. from a stock master list)
            # For now, using symbol as keyword if company_name is not readily available.
            # The NewsAnalysisAgent needs to be modified to accept end_date_override.
            analysis = self.news_analysis_agent.analyze_news_contrarian(
                keywords=company_name, # Or just symbol if name not available
                symbols_to_correlate=[symbol],
                days_history_for_correlation=5,
                days_news=7, # News up to 7 days before current_date
                limit_news=10,
                end_date_override=current_date # This is the new parameter
            )
            if isinstance(analysis, str): # Error string from agent
                return {"error": analysis, "raw_output": analysis}
            return analysis
        except Exception as e:
            print(f"Error in _get_point_in_time_news_analysis for {symbol} on {current_date}: {e}")
            return {"error": str(e)}

    def _get_point_in_time_technical_analysis(self, symbol: str, current_date: datetime.date) -> Dict:
        """
        Fetches technical analysis for a symbol.
        NOTE: Current TechnicalAnalysisAgent.analyze does not support point-in-time analysis. This is an approximation.
        In a real scenario, historical data up to current_date would be fed to it.
        """
        print(f"[{current_date}] Getting technical analysis for {symbol}...")
        try:
            # This is an approximation. The TA agent should ideally analyze data *up to* current_date.
            # The current TA agent's analyze method might use current real-time data, which is not ideal for backtesting.
            # For a proper backtest, this method would need to be refactored or the TA agent modified.
            analysis_str = self.technical_analysis_agent.analyze(symbol=symbol) 
            # Attempt to parse if it's stringified JSON, or wrap it
            try:
                # The analyze method in TechnicalAnalysisAgent returns a string, which might be a JSON string or just text.
                # For consistency, we'll try to ensure it's a dict.
                if isinstance(analysis_str, str):
                    try:
                        # Check if it's already a JSON string
                        parsed_analysis = json.loads(analysis_str)
                        return parsed_analysis
                    except json.JSONDecodeError:
                        # If not a JSON string, wrap it in a dict
                        return {"analysis_summary": analysis_str}
                elif isinstance(analysis_str, dict):
                     return analysis_str # Already a dict
                else:
                    return {"error": "Unknown TA output format", "raw_output": str(analysis_str)}

            except Exception as parse_e:
                 print(f"Error parsing TA for {symbol} on {current_date}: {parse_e}")
                 return {"error": f"Failed to parse TA output: {parse_e}", "raw_output": analysis_str}

        except Exception as e:
            print(f"Error in _get_point_in_time_technical_analysis for {symbol} on {current_date}: {e}")
            return {"error": str(e)}

    def _get_point_in_time_market_forecast(self, symbol: str, current_date: datetime.date) -> Dict:
        """
        Returns a mock market forecast for backtesting.
        """
        print(f"[{current_date}] Getting mock market forecast for {symbol}...")
        return {
            "symbol_analyzed": symbol,
            "overall_sentiment": "Neutral", # Could be randomized or based on some macro indicator later
            "trend_prediction": "Sideways",
            "confidence": 0.5,
            "reasoning": f"Mock historical forecast for {symbol} on {current_date} for backtesting purposes.",
            "forecast_date": str(current_date)
        }

    def _get_point_in_time_financial_summary(self, symbol: str, current_date: datetime.date) -> Optional[Dict]:
        """
        Returns a mock financial summary for backtesting.
        In a real scenario, this would fetch the latest financials available *before* current_date.
        """
        print(f"[{current_date}] Getting mock financial summary for {symbol}...")
        # This should ideally fetch actual historical fundamental data from a database/API
        # For now, returning a static mock object.
        return {
            "P/E": 15.0 + (hash(symbol) % 10 - 5), # Add some variation
            "P/B": 1.5 + (hash(symbol) % 10 / 10 - 0.5),
            "Debt/Equity": 0.5 + (hash(symbol) % 5 / 10 - 0.2),
            "retrieved_date": str(current_date),
            "notes": "Mock financial data for backtesting."
        }

    def _get_point_in_time_persona_analyses(self, symbol: str, company_info: Dict, 
                                           financials_summary: Optional[Dict], 
                                           contrarian_news_analysis: Dict, 
                                           market_forecast: Dict, 
                                           current_date: datetime.date) -> List[Dict]:
        """
        Gathers analyses from specialized persona agents.
        """
        print(f"[{current_date}] Getting persona analyses for {symbol}...")
        analyses = []
        try:
            if not financials_summary: # Ensure financials_summary is not None
                financials_summary = {"notes": "Financial summary not available for persona analysis."}

            # Deep Value Contrarian Persona
            dv_analysis = self.deep_value_agent.analyze_opportunity(
                symbol=symbol,
                company_info=company_info,
                financials_summary=financials_summary,
                contrarian_sentiment_analysis=contrarian_news_analysis,
                market_forecast=market_forecast.get("trend_prediction") 
            )
            analyses.append(dv_analysis if isinstance(dv_analysis, dict) else {"error": "DV Agent failed", "raw": str(dv_analysis)})
            
            # Market Manipulation Analyst Persona
            # This agent might need a summary of recent price/volume, which we don't have readily here.
            # We'll pass None for now, or a mock summary.
            price_volume_mock = f"Mock price/volume summary for {symbol} up to {current_date}"
            mm_analysis = self.manipulation_spotter_agent.assess_manipulation(
                symbol=symbol,
                contrarian_sentiment_analysis=contrarian_news_analysis,
                recent_price_volume_summary=price_volume_mock
            )
            analyses.append(mm_analysis if isinstance(mm_analysis, dict) else {"error": "MM Agent failed", "raw": str(mm_analysis)})

        except Exception as e:
            print(f"Error in _get_point_in_time_persona_analyses for {symbol} on {current_date}: {e}")
            analyses.append({"error": f"Persona analysis failed: {str(e)}"})
        return analyses

    def _simulate_trade(self, trade_details: Dict, trade_price: float, current_date: datetime.date):
        """
        Simulates a trade and updates the portfolio.
        Assumes trade_details contains 'symbol', 'action', 'quantity'.
        """
        symbol = trade_details['symbol']
        action = trade_details['action'].upper()
        quantity = trade_details.get('quantity') # This needs to be resolved from quantity_suggestion

        if quantity is None: # Try to parse from quantity_suggestion
            qty_suggestion = trade_details.get('quantity_suggestion', "0 shares")
            try:
                quantity = int(str(qty_suggestion).split(" ")[0])
            except ValueError:
                print(f"[{current_date}] Error: Could not parse quantity from suggestion '{qty_suggestion}' for {symbol}. Skipping trade.")
                return
        
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            print(f"[{current_date}] Error: Invalid quantity {quantity} for {symbol}. Skipping trade.")
            return

        cost = quantity * trade_price

        print(f"[{current_date}] Simulating Trade: {action} {quantity} {symbol} @ {trade_price:,.2f}, Cost: {cost:,.2f}")

        if action == "BUY":
            if self.portfolio["cash"] < cost:
                print(f"[{current_date}] Warning: Insufficient cash to buy {quantity} {symbol}. Cash: {self.portfolio['cash']:.2f}, Cost: {cost:.2f}. Skipping trade.")
                return
            
            self.portfolio["cash"] -= cost
            if symbol in self.portfolio["holdings"]:
                current_qty = self.portfolio["holdings"][symbol]["quantity"]
                current_avg_cost = self.portfolio["holdings"][symbol]["average_cost"]
                new_total_qty = current_qty + quantity
                new_total_cost_basis = (current_qty * current_avg_cost) + cost
                self.portfolio["holdings"][symbol]["average_cost"] = new_total_cost_basis / new_total_qty
                self.portfolio["holdings"][symbol]["quantity"] = new_total_qty
            else:
                self.portfolio["holdings"][symbol] = {"quantity": quantity, "average_cost": trade_price}
        
        elif action == "SELL":
            if symbol not in self.portfolio["holdings"] or self.portfolio["holdings"][symbol]["quantity"] == 0:
                print(f"[{current_date}] Warning: No holdings of {symbol} to sell. Skipping trade.")
                return
            if quantity > self.portfolio["holdings"][symbol]["quantity"]:
                print(f"[{current_date}] Warning: Attempting to sell {quantity} {symbol}, but only hold {self.portfolio['holdings'][symbol]['quantity']}. Selling available amount.")
                quantity = self.portfolio["holdings"][symbol]["quantity"]
                cost = quantity * trade_price # Recalculate cost for actual sold quantity
            
            self.portfolio["cash"] += cost # Technically, this is proceeds, not cost.
            self.portfolio["holdings"][symbol]["quantity"] -= quantity
            if self.portfolio["holdings"][symbol]["quantity"] == 0:
                del self.portfolio["holdings"][symbol] # Remove if all sold
        
        else:
            print(f"[{current_date}] Unknown trade action: {action} for {symbol}. Skipping.")
            return

        self.trade_log.append({
            "date": str(current_date), 
            "symbol": symbol, 
            "action": action, 
            "quantity": quantity, 
            "price": trade_price, 
            "total_transaction_value": cost # For BUY this is cost, for SELL this is proceeds
        })

    def run_backtest(self):
        """
        Runs the backtesting simulation.
        """
        print("Starting backtest run...")
        # Fetch all historical data first
        for symbol in self.stock_symbols:
            print(f"Fetching historical data for {symbol}...")
            # AKShare uses YYYYMMDD format for dates
            hist_df = akshare_utils.get_stock_history(
                symbol=symbol, 
                start_date=self.start_date.strftime("%Y%m%d"),
                end_date=self.end_date.strftime("%Y%m%d"),
                adjust="hfq" # Using后复权 (hfq) for price adjustments
            )
            if not hist_df.empty:
                # Ensure the index is datetime
                if not isinstance(hist_df.index, pd.DatetimeIndex):
                     hist_df.index = pd.to_datetime(hist_df.index)
                self.historical_price_data[symbol] = hist_df
            else:
                print(f"Warning: Could not fetch historical data for {symbol}. It will be excluded.")
        
        if not self.historical_price_data:
            print("Error: No historical price data fetched. Aborting backtest.")
            return

        # Create date range for simulation
        # Using pandas date_range for business days might be more realistic
        sim_date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

        for current_sim_date_dt in sim_date_range:
            current_sim_date = current_sim_date_dt.date() # Convert pandas Timestamp to datetime.date
            print(f"\nProcessing Date: {current_sim_date}")

            for symbol in self.stock_symbols:
                if symbol not in self.historical_price_data:
                    continue

                symbol_hist_data = self.historical_price_data[symbol]
                
                # Get price data for the current simulation date
                # Ensure index is datetime.date or comparable
                try:
                    # If index is DatetimeIndex, convert current_sim_date to Timestamp for exact match
                    current_day_price_data = symbol_hist_data.loc[pd.Timestamp(current_sim_date)]
                except KeyError:
                    # print(f"No price data for {symbol} on {current_sim_date}. Skipping symbol for this day.")
                    continue # Skip if no data for this day (e.g. market holiday not caught by freq='B', or stock not listed yet)
                
                if pd.isna(current_day_price_data.get('收盘')): # AKShare uses '收盘'
                    print(f"Close price is NaN for {symbol} on {current_sim_date}. Skipping.")
                    continue
                
                trade_price = current_day_price_data['收盘'] # Use closing price for trade execution

                # Simplified company_info - in a real system, this would be richer
                # Potentially fetch from ak.stock_individual_info_em once per symbol if needed
                company_info = {"symbol": symbol, "name": symbol, "current_price": trade_price} 
                
                # --- Gather all analyses for this symbol and date ---
                news_analysis = self._get_point_in_time_news_analysis(symbol, company_info['name'], current_sim_date)
                tech_analysis = self._get_point_in_time_technical_analysis(symbol, current_sim_date)
                market_fcst = self._get_point_in_time_market_forecast(symbol, current_sim_date)
                fin_summary = self._get_point_in_time_financial_summary(symbol, current_sim_date)
                
                persona_analyses = self._get_point_in_time_persona_analyses(
                    symbol, company_info, fin_summary, 
                    news_analysis, market_fcst, current_sim_date
                )

                # --- Make trade decision using PortfolioManagerAgent ---
                print(f"[{current_sim_date}] Generating trade decision for {symbol}...")
                pm_decision_output = self.portfolio_manager_agent.generate_trade_decision_contrarian(
                    symbol=symbol,
                    market_forecast=market_fcst,
                    technical_analysis=tech_analysis,
                    contrarian_sentiment_analysis=news_analysis,
                    persona_analyses=persona_analyses,
                    financial_report_summary=fin_summary,
                    current_portfolio_summary=self.portfolio # Pass current portfolio state
                )

                if isinstance(pm_decision_output, str): # Error occurred
                    print(f"[{current_sim_date}] Error from PortfolioManagerAgent for {symbol}: {pm_decision_output}")
                    continue
                
                pm_decision_json = pm_decision_output # It's already a dict

                final_decision_block = pm_decision_json.get("final_decision_after_risk", {})
                trade_action = final_decision_block.get("decision")
                
                if trade_action in ["BUY", "SELL"]:
                    trade_details = final_decision_block.get("proposed_trade_details")
                    if trade_details and isinstance(trade_details, dict):
                        # Ensure trade_details has symbol and action, compatible with _simulate_trade
                        trade_details_for_sim = {
                            "symbol": symbol, # From outer loop
                            "action": trade_action,
                            "quantity_suggestion": trade_details.get("quantity_suggestion"), # For parsing in _simulate_trade
                            "quantity": trade_details.get("quantity") # If PortfolioManager directly provides it
                        }
                        self._simulate_trade(trade_details_for_sim, trade_price, current_sim_date)
                    else:
                        print(f"[{current_sim_date}] Warning: No valid trade_details for {trade_action} on {symbol}. Decision: {pm_decision_json}")
            
            # --- Update daily portfolio value ---
            current_total_value = self.portfolio["cash"]
            for sym, details in self.portfolio["holdings"].items():
                if sym in self.historical_price_data:
                    sym_hist_data = self.historical_price_data[sym]
                    try:
                        current_sym_price = sym_hist_data.loc[pd.Timestamp(current_sim_date)]['收盘']
                        if pd.notna(current_sym_price):
                            current_total_value += details["quantity"] * current_sym_price
                        # else: price not available for this day, holding value not updated with current price
                    except KeyError:
                        # Price not available for this day, use last known average cost for this calculation if needed, or skip
                        # For simplicity, if price is not there, we are not adding its value from holdings to current_total_value for this day's snapshot.
                        # A more robust approach might carry forward the last known price or use average cost.
                        pass 
            
            self.portfolio["total_value"] = current_total_value # Update portfolio's total value
            self.daily_portfolio_value.append({"date": str(current_sim_date), "value": current_total_value})
            print(f"[{current_sim_date}] Portfolio Value: ${current_total_value:,.2f}")

        print("\nBacktest run completed.")
        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculates basic performance metrics for the backtest.
        """
        print("\nCalculating Performance Metrics...")
        metrics = {}
        
        if not self.daily_portfolio_value:
            metrics["total_return_pct"] = 0.0
            metrics["final_portfolio_value"] = self.initial_capital
        else:
            final_value = self.daily_portfolio_value[-1]["value"]
            metrics["final_portfolio_value"] = final_value
            metrics["total_return_pct"] = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
        metrics["number_of_trades"] = len(self.trade_log)
        
        # Placeholder for future metrics
        metrics["sharpe_ratio"] = "N/A (Not Implemented)"
        metrics["max_drawdown"] = "N/A (Not Implemented)"

        print("--- Performance Metrics ---")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Number of Trades: {metrics.get('number_of_trades', 0)}")
        print("---------------------------")
        
        # Optionally, save logs
        # with open("trade_log.json", "w") as f:
        #     json.dump(self.trade_log, f, indent=2)
        # with open("daily_portfolio_value.json", "w") as f:
        #     json.dump(self.daily_portfolio_value, f, indent=2)
            
        return metrics

# Example Usage (Illustrative - will not run without proper setup and data)
# if __name__ == "__main__":
#     # Mock LLM config
#     llm_config_test = {
#         "config_list": [{"model": "gpt-4", "api_key": "YOUR_API_KEY"}], # Replace with actual config
#         "cache_seed": 42 
#     }
#     # Ensure get_llm_config_for_autogen is correctly imported or llm_config is directly usable by agents
    
#     backtester = Backtester(
#         llm_config=llm_config_test, # This needs to be a valid config for autogen agents
#         start_date_str="2023-01-01",
#         end_date_str="2023-03-31", # Shorter period for testing
#         initial_capital=100000.0,
#         stock_symbols=["000001", "600519"] # Example: Ping An Bank, Kweichow Moutai
#     )
#     results = backtester.run_backtest()
#     print("\nBacktest Results:")
#     for key, value in results.items():
#         print(f"{key}: {value}")
