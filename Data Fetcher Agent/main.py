from graph import build_graph

def run_agent():

    graph = build_graph()

    initial_state = {
        "datasets": [],

        "customers_data": None,
        "transactions_data": None,
        "accounts_data": None,

        "metrics": {},
        "logs": []
    }

    result = graph.invoke(initial_state)

    print("\nRegulatory Metrics:\n")
    print(result["metrics"])

    print("\nExecution Logs:\n")

    for log in result["logs"]:
        print(log)


if __name__ == "__main__":
    run_agent()