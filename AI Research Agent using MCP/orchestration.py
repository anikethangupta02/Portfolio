import asyncio
from dotenv import load_dotenv

from agents.researcher import ResearchAgent
from agents.planner import PlannerAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent

load_dotenv()

async def main():
    task = "Explain Transformer models for beginners"

    planner = PlannerAgent()
    researcher = ResearchAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()

    plan = planner.plan(task)
    print("PLAN:", plan)

    research = await researcher.run(task)
    print("RESEARCH:", research)

    draft = writer.write(research)
    final = reviewer.review(draft)

    print("\nFINAL ANSWER:\n")
    print(final)


if __name__ == "__main__":
    asyncio.run(main())