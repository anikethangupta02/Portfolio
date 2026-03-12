import pandas as pd

def data_fetch_node(state):

    for dataset in state["datasets"]:

        try:

            df = pd.read_csv(dataset)

            if "customers" in dataset:
                state["customers_data"] = df

            elif "transactions" in dataset:
                state["transactions_data"] = df

            elif "accounts" in dataset:
                state["accounts_data"] = df

            state["logs"].append(
                f"{dataset} loaded successfully with {len(df)} rows"
            )

        except Exception as e:

            state["logs"].append(
                f"Error loading {dataset}: {str(e)}"
            )

    return state