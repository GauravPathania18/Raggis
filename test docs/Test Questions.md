# RAG System Evaluation — Test Question Bank

> Structured across three difficulty levels to evaluate retrieval quality, semantic understanding, and cross-document reasoning.

---

## Overview

This document defines the formal test suite used to benchmark APEX and baseline RAG systems. Questions are stratified into three levels of increasing cognitive demand, modelled on Bloom's Taxonomy: factual recall (Level 1), conceptual understanding (Level 2), and analytical synthesis (Level 3).

**Source documents:**
- *The Road to React* — Robin Wieruch (2024)
- *The Book of Five Rings* — Miyamoto Musashi

Each level targets a distinct capability in the retrieval pipeline. Level 1 validates precise token-level lookup; Level 2 probes semantic chunking and relational reasoning; Level 3 stress-tests multi-document synthesis and contextual inference.

---

## Level 1 — Simple Factual Questions

**Difficulty:** Low | **Skill:** Direct Retrieval | **Target metric:** Token F1

These questions have a single, unambiguous answer that should appear verbatim or near-verbatim in the source text. A high-quality RAG system is expected to achieve Token F1 ≥ 0.90 on this tier. Failure here indicates a fundamental retrieval failure — the system cannot locate the correct chunk.

| # | Question | Expected Answer | Source |
|---|----------|-----------------|--------|
| 1 | Who wrote *The Road to React*? | Robin Wieruch | The Road to React |
| 2 | What is the publication date of *The Road to React*? | 2024-02-05 (February 5, 2024) | The Road to React |
| 3 | Who authored *The Book of Five Rings*? | Miyamoto Musashi | The Book of Five Rings |

**Evaluation note:** These questions serve as a sanity check. If a retrieval system fails here, the issue lies in embedding coverage or chunking granularity rather than reasoning ability. Formally, if the ground-truth answer is the token set $A$ and the retrieved answer is $R$, then:

$$\text{Token F1} = \frac{2 \cdot |A \cap R|}{|A| + |R|}$$

This should approach 1.0 for all three questions.

---

## Level 2 — Conceptual & Relational Questions

**Difficulty:** Medium | **Skill:** Semantic Reasoning | **Target metric:** Context Relevance

These questions require the system to connect concepts across multiple chunks and demonstrate semantic understanding beyond surface-level string matching. Evaluation shifts from Token F1 to Context Relevance and Hallucination Rate. Acceptable thresholds: Context Relevance ≥ 0.75, Hallucination Rate ≤ 0.10.

---

### Q4 — What is JSX and why is it used in React?

**Source:** The Road to React

**Expected answer:**

JSX is a JavaScript syntax extension enabling HTML-like markup inside component code. Key reasons for its use:
- Improves component readability and structure
- Bridges declarative UI design with JavaScript logic
- Transpiles to `React.createElement()` calls at build time

---

### Q5 — According to *The Book of Five Rings*, what is the "Way of Strategy"?

**Source:** The Book of Five Rings

**Expected answer:**

A path of continuous self-improvement and principled adaptability. Core tenets:
- Mastery of fundamentals precedes advanced technique
- Practical experience over rote memorisation
- Situational awareness of both self and opponent
- Applicability beyond combat — to all disciplines of life

---

### Q6 — What are React Hooks and why were they introduced?

**Source:** The Road to React

**Expected answer:**

Hooks are functions that bring state and lifecycle features into functional components. Motivation:
- Eliminated the complexity of class-based components
- Enabled reuse of stateful logic without render props or HOCs
- Produced more composable, testable component code

---

**Evaluation note:** Context Relevance is computed as the fraction of retrieved context passages that are semantically necessary for answering the question:

$$\text{Context Relevance} = \frac{\text{# relevant chunks retrieved}}{\text{total chunks retrieved}}$$

A score below 0.75 indicates over-retrieval or noisy reranking.

---

## Level 3 — Analytical & Cross-Document Questions

**Difficulty:** High | **Skill:** Synthesis & Inference | **Target metric:** Multi-Doc QA Score

These questions require the system to synthesise evidence from multiple documents and construct novel connections that do not exist verbatim in any single source. They represent the hardest tier of the benchmark, targeting the Memory Manager and Query Intelligence module in APEX. Expected Multi-Doc QA Score ≥ 0.70; Hallucination Rate ≤ 0.05.

---

### Q7 — Philosophical Parallels in Mastery

> **Question:** Compare the philosophical approach to mastery in *The Book of Five Rings* with the learning approach to mastering React. What similarities do they share?

**Sources required:** Both documents

**Expected answer — shared principles:**

- **Fundamentals before complexity** — Both stress that beginners must internalise core building blocks (JavaScript mechanics; the five rings) before layering advanced techniques.
- **Learning through doing** — Musashi's insistence on real combat experience mirrors React's emphasis on shipping actual components over reading theory.
- **Principles over pattern-matching** — Understanding *why* a hook avoids re-renders parallels understanding *why* timing an attack exploits an opening.
- **Iterative refinement** — The warrior trains daily; the developer ships, measures, and refactors continuously.
- **Contextual adaptability** — Neither source prescribes rigid recipes; both demand situational judgment.

---

### Q8 — The Void and Clean Code

> **Question:** How does the concept of "emptiness" (void) in Musashi's philosophy relate to writing clean, maintainable React code?

**Sources required:** Both documents

**Expected answer — mapping void to code quality:**

- **Clarity of mind without obstruction** → Single-responsibility components that do one thing and do it well.
- **Absence of unnecessary thought** → No speculative abstractions; no premature generalisation.
- **Clean separation of concerns** → Data-fetching, state management, and rendering kept in distinct layers.
- **Removing what is unused** → Dead code and orphaned dependencies purged, like eliminating unnecessary motion before a strike.
- **Effortless readability** → Code whose intent is immediately apparent requires no inline explanation, as mastery requires no explanation of technique.

---

### Q9 — Musashi's Advice to the React Developer

> **Question:** Based on both books, what advice would Musashi give to a React developer about mastering their craft?

**Sources required:** Primarily *The Book of Five Rings*; grounded in *The Road to React*

**Expected answer — seven Musashi-derived principles:**

| Musashi's Maxim | React Translation |
|-----------------|-------------------|
| **Know the fundamentals** | Master JavaScript — closures, the event loop, promises — before touching any framework. |
| **Practice daily** | Build something small every day; consistent deliberate practice compounds faster than weekend sprints. |
| **Study many paths** | Explore multiple state-management approaches (Context, Zustand, Redux) to understand trade-offs, not just defaults. |
| **Empty your mind** | Approach each new paradigm (Server Components, concurrent rendering) without preconceptions from older patterns. |
| **See what cannot be seen** | Read the underlying diffing algorithm and reconciler; understand what the framework does invisibly. |
| **Do nothing unnecessary** | Write minimal, purposeful code. If a component, hook, or dependency cannot be justified, remove it. |
| **The Way is in training** | Ship real projects with real users. No tutorial substitutes for production constraints. |

---

## Benchmark Summary

The table below maps each level to its primary evaluation metric and the minimum acceptable threshold for a system to be considered competitive against baseline RAG approaches.

| Level | Capability Tested | Primary Metric | Threshold |
|-------|-------------------|----------------|-----------|
| 1 | Direct Retrieval | Token F1 = $2\|A\cap R\| / (\|A\|+\|R\|)$ | ≥ 0.90 |
| 2 | Semantic Reasoning | Context Relevance (fraction of useful chunks) | ≥ 0.75 |
| 3 | Cross-Doc Synthesis | Multi-Doc QA Score + Hallucination Rate | ≥ 0.70 / ≤ 0.05 |

---

*APEX Evaluation Framework · Internal Use*
