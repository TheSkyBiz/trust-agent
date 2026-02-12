import re
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# --------------------------------------------------
# MODEL SETUP (ASYMMETRIC ARCHITECTURE)
# --------------------------------------------------

answer_llm = ChatOllama(
    model="qwen2.5:3b",     # fast generator
    temperature=0.3,
    num_ctx=2048,
    num_predict=256,
    keep_alive="10m"
)

critic_llm = ChatOllama(
    model="llama3:8b",      # stronger evaluator
    temperature=0.1,
    num_ctx=2048,
    num_predict=200,
    keep_alive="10m"
)


# --------------------------------------------------
# SYSTEM PROMPTS
# --------------------------------------------------

ANSWER_SYSTEM = SystemMessage(content="""
You are a confident and helpful AI assistant.
Provide clear, accurate answers.
If the question contains false assumptions, correct them.
Keep responses under 150 words.
""")

CRITIC_SYSTEM = SystemMessage(content="""
You are a strict AI evaluator.

Evaluate the assistant's response for:

- hallucination risk
- logical errors
- factual accuracy
- overconfidence

Return EXACTLY in this format:

Trust Score: <number>/10
Verdict: TRUST / CAUTION / DO NOT TRUST
Reason: <concise explanation under 80 words>

Be objective and extremely concise.
""")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def extract_score(text: str):
    """Extract trust score from critic output."""
    match = re.search(r"(\d+)/10", text)
    return int(match.group(1)) if match else None


def log_run(question, answer, critique, regenerated=None):
    """Append run data for lightweight evaluation tracking."""
    with open("runs.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"QUESTION:\n{question}\n\n")
        f.write(f"ANSWER:\n{answer}\n\n")
        f.write(f"CRITIC:\n{critique}\n\n")

        if regenerated:
            f.write(f"REGENERATED ANSWER:\n{regenerated}\n\n")


def regenerate_answer(question):
    """Safer second-pass generation."""
    safer_prompt = SystemMessage(content="""
Provide a careful and factual response.
Admit uncertainty when appropriate.
Avoid speculation.
Keep it concise.
""")

    response = answer_llm.invoke([
        safer_prompt,
        HumanMessage(content=question)
    ])

    return response.content


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

print("\n✅ TrustAgent Initialized")
print("Type 'exit' to quit.\n")

while True:

    question = input("Ask something: ")

    if question.lower() == "exit":
        print("\nGoodbye 👋")
        break

    # ---------- GENERATE ----------
    answer_msg = answer_llm.invoke([
        ANSWER_SYSTEM,
        HumanMessage(content=question)
    ])

    answer = answer_msg.content

    print("\n🤖 Answer Agent:\n")
    print(answer)

    # ---------- CRITIQUE ----------
    critique_msg = critic_llm.invoke([
        CRITIC_SYSTEM,
        HumanMessage(
            content=f"Question: {question}\n\nAssistant Answer: {answer}"
        )
    ])

    critique = critique_msg.content

    print("\n🧠 Critic Agent:\n")
    print(critique)

    # ---------- DECISION ----------
    score = extract_score(str(critique))
    regenerated_answer = None

    if score is not None:

        if score < 5 and "DO NOT TRUST" in critique:
            print("\n🚨 Unsafe answer detected → Regenerating...\n")

            regenerated_answer = regenerate_answer(question)

            print("✅ Regenerated Answer:\n")
            print(regenerated_answer)

        elif score < 8:
            print("\n⚠️ Use caution.\n")

        else:
            print("\n✅ Response appears reliable.\n")

    else:
        print("\n⚠️ Could not parse trust score.\n")

    # ---------- LOG ----------
    log_run(question, answer, critique, regenerated_answer)
