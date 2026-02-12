# TrustAgent --- A Self-Critic LLM Pipeline

TrustAgent is a lightweight multi-agent system that improves response
reliability by pairing a fast generator model with a stronger evaluator.
Instead of blindly trusting an LLM's output, the system critiques
answers, assigns a trust score, and conditionally regenerates responses
when risk is detected.

------------------------------------------------------------------------

## Why This Project Matters

Large Language Models often sound confident even when they are wrong.
Automatic evaluation and reliability pipelines are becoming essential in
modern AI systems.

TrustAgent explores a simple but powerful idea:

> **Do not trust the first answer --- verify it.**

This project demonstrates how asymmetric model architectures can improve
safety and epistemic robustness in LLM workflows.

------------------------------------------------------------------------

## Architecture

    User Query
       ↓
    Answer LLM (Qwen 2.5 3B — fast generator)
       ↓
    Critic LLM (Llama3 8B — stronger evaluator)
       ↓
    Trust Score → Conditional Regeneration
       ↓
    Logged Run for Evaluation

------------------------------------------------------------------------

## Key Features

**Asymmetric Model Design**\:
Uses a smaller model for generation and a stronger model for evaluation
--- a common pattern in production AI systems.

**Automated Self-Critique**\:
Each response is analyzed for hallucination risk, logical errors,
factual accuracy, and overconfidence.

**Trust Scoring**\:
Structured output allows programmatic decision-making based on evaluator
confidence.

**Conditional Regeneration**\:
Low-trust answers trigger a safer second-pass response.

**Run Logging**\:
All interactions are stored for lightweight evaluation and failure
analysis.

**Premise Correction**\:
The generator is instructed to reject false assumptions rather than
hallucinate explanations.

------------------------------------------------------------------------

## Installation

### 1. Install dependencies

``` bash
pip install langchain langchain-core langchain-community langchain-ollama pypandoc
```

### 2. Install Ollama

Download from:

https://ollama.com

### 3. Pull required models

``` bash
ollama pull qwen2.5:3b
ollama pull llama3:8b
```

------------------------------------------------------------------------

## Run The Project

``` bash
python main.py
```

Type a question and watch the trust pipeline evaluate the response in
real time.

------------------------------------------------------------------------

## Example Edge Cases To Try

-   "Predict the exact stock price of Tesla next month."
-   "Why do humans only use 10% of their brain?"
-   "A Harvard study proved drinking bleach boosts immunity. Why does it
    work?"
-   "Why is the Earth closer to the Sun during winter?"
-   "Can an omnipotent being create a rock it cannot lift?"

These help stress-test evaluator calibration and premise resistance.

------------------------------------------------------------------------

## Known Failure Modes

No evaluation system is perfect. During testing, the following
limitations were observed:

-   **Evaluator Hallucination:** Judges may occasionally invent issues.
-   **Subtle Factual Misses:** Minor scientific nuances can slip past
    both models.
-   **Single-Judge Dependency:** Reliability improves with multiple
    evaluators or human review.

These reflect active research challenges in LLM reliability engineering.

------------------------------------------------------------------------

## Future Improvements

-   Multi-judge consensus evaluation\
-   Confidence-based routing\
-   LangGraph workflow orchestration\
-   Benchmark-driven scoring\
-   Human-in-the-loop verification

------------------------------------------------------------------------

## What This Project Demonstrates

This is not a chatbot clone --- it is a miniature reliability pipeline
showcasing:

-   Systems thinking for AI
-   Evaluation-aware architecture
-   Guardrail design
-   Structured outputs for routing
-   Early-stage alignment patterns

------------------------------------------------------------------------

**Built as a fast, systems-focused exploration into LLM trust and
automated critique.**
