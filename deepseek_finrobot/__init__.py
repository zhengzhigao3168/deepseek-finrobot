"""
DeepSeek FinRobot - A financial analysis and decision-making framework.
"""

__version__ = "0.1.0"

# Import key classes from submodules to make them available at the top level of the package
from .agents import (
    MarketForecasterAgent,
    FinancialReportAgent,
    NewsAnalysisAgent,
    IndustryAnalysisAgent,
    PortfolioManagerAgent,
    TechnicalAnalysisAgent,
    RiskManagerAgent,
    DeepValueContrarianPersonaAgent,
    MarketManipulationAnalystPersonaAgent,
    SingleAssistant,
    SingleAssistantShadow,
    MultiAgentWorkflow
)

from .data_source import (
    akshare_utils,
    cn_news_utils
    # Add other utils if they become primary interfaces
)

from .utils import (
    get_current_date,
    format_financial_number,
    cached,
    get_llm_config_for_autogen
    # Add other common utils
)

# Backtester - if it's intended to be a primary interface
try:
    from .backtester import Backtester
    _BACKTESTER_AVAILABLE = True
except ImportError as e:
    _BACKTESTER_AVAILABLE = False
    print(f"Warning: Backtester module not found or has missing dependencies: {e}")


__all__ = [
    # Agents
    'MarketForecasterAgent',
    'FinancialReportAgent',
    'NewsAnalysisAgent',
    'IndustryAnalysisAgent',
    'PortfolioManagerAgent',
    'TechnicalAnalysisAgent',
    'RiskManagerAgent',
    'DeepValueContrarianPersonaAgent',
    'MarketManipulationAnalystPersonaAgent',
    'SingleAssistant',
    'SingleAssistantShadow',
    'MultiAgentWorkflow',

    # Data Source Utilities
    'akshare_utils',
    'cn_news_utils',

    # General Utilities
    'get_current_date',
    'format_financial_number',
    'cached',
    'get_llm_config_for_autogen',
    
    '__version__',
]

if _BACKTESTER_AVAILABLE:
    __all__.append('Backtester')
