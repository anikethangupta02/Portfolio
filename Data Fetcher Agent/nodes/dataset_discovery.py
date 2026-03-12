def dataset_discovery(state):

    state["datasets"] = [
        "data/customers.csv",
        "data/transactions.csv",
        "data/accounts.csv"
    ]

    state["logs"].append("Datasets identified for regulatory reporting")

    return state