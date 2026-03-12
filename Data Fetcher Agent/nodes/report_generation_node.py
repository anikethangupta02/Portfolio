import json
from datetime import datetime

def report_generation_node(state):

    report = {
        "report_type": "AML_KYC_Compliance",
        "generated_at": str(datetime.utcnow()),
        "metrics": state["metrics"]
    }

    with open("regulatory_output.json", "w") as f:
        json.dump(report, f, indent=4)

    state["logs"].append("Regulatory report generated")

    return state