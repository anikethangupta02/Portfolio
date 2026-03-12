from typing import TypedDict, Optional
import pandas as pd

class AgentState(TypedDict):
    datasets: list

    customers_data: Optional[pd.DataFrame]
    transactions_data: Optional[pd.DataFrame]
    accounts_data: Optional[pd.DataFrame]

    metrics: dict
    logs: list