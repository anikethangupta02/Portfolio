def regulatory_metrics_node(state):

    customers = state["customers_data"]
    transactions = state["transactions_data"]
    accounts = state["accounts_data"]

    metrics = {}

    metrics["total_transactions"] = len(transactions)

    metrics["flagged_transactions"] = int(
        transactions[transactions["flagged"] == True].shape[0]
    )

    metrics["total_transaction_volume"] = float(
        transactions["amount"].sum()
    )

    metrics["pending_kyc_customers"] = int(
        customers[customers["kyc_status"] == "Pending"].shape[0]
    )

    metrics["high_risk_customers"] = int(
        customers[customers["risk_rating"] == "High"].shape[0]
    )

    metrics["total_accounts"] = len(accounts)

    metrics["average_account_balance"] = float(
        accounts["balance"].mean()
    )

    state["metrics"] = metrics

    state["logs"].append("Regulatory metrics computed")

    return state