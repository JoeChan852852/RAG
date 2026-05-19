# 🧠 Module 3 Tutorial Session (60–90 min): **AI as Your Professional Assistant**

## **Technical Reading & Writing with AI (LLMs)**


## 🧭 Session Objectives (what students can do by the end)

By the end of this session, students will be able to:

1. **Summarize** a real engineering research paper quickly using an LLM.
2. **Compare two papers** side-by-side (methods, results, limits, strengths/weaknesses).
3. **Extract key technical info** (equations, datasets, assumptions, implementation details) from papers + forums.
4. **Critically evaluate AI output** (spot missing evidence, errors, made-up details).
5. Use best practices to **reduce hallucinations** and **increase reliability**.

***

# 🧩 Session Plan (60–90 min, interactive)

|  Time | Segment                                 | What happens                                 |
| ----: | --------------------------------------- | -------------------------------------------- |
|  0–10 | 1) Hook + Why this matters              | Story + quick demo result vs. messy reading  |
| 10–30 | 2) Core Skill A: Summarize one paper    | Prompt templates + “summary with evidence”   |
| 30–50 | 3) Core Skill B: Compare two papers     | Side-by-side table + “method/results/limits” |
| 50–65 | 4) Extract details from papers + forums | Equations, assumptions, code hints, pitfalls |
| 65–80 | 5) Hallucination traps + best practices | Warning box + verification workflow          |
| 80–90 | 6) Mini-challenge + wrap-up             | 1-page brief + cheat sheet + integrity       |

> ✅ If you only have **60 minutes**, skip the “Tool Strategy” deep dive and shorten the mini-challenge.

***

# 1) Introduction (Hook + Motivation) — 0–10 min ⚡

### 🎯 Hook scenario (relatable)

> **“It’s 10pm.** You have **two fresh research papers** on battery thermal management due tomorrow.  
> You also have a lab report and a quiz.  
> You *could* read both papers line-by-line…  
> Or you can use AI to get **a correct first-pass** in **15 minutes**, then verify the important bits.”

### Why this skill is *high-leverage* in the AI era

Technical reading/writing is basically your engineering “multiplier”:

* **Reading faster** → you learn faster → you design better solutions
* **Writing clearer** → your ideas get accepted more often (projects, internships, research)
* **AI helps you do the boring part** (structure, summarizing, drafting), so you can focus on the **thinking part** ✅

**Key mindset:**

> AI is a **power tool**, not a brain replacement.  
> You still own the final answer.

***

# 2) Core Skills to Teach  — 10–50 min

## Skill A — Summarize **one** technical paper (fast + deep)

### ✅ Goal

Produce a summary that answers:

* **What problem?**
* **What method?**
* **What results?**
* **What assumptions?**
* **What limitations?**
* **So what? (engineering takeaway)**

### 🧠 The “5-Box Summary” (super scannable)

Use this structure every time:

1. **Problem & context (2–3 lines)**
2. **Approach (what they did)**
3. **Key results (numbers, graphs, comparisons)**
4. **Assumptions & limits (what must be true?)**
5. **Practical takeaway (what you can reuse)**

> Tip: If the summary has **no numbers**, it’s usually too fluffy.

***

## Skill B — Compare **two** papers side-by-side

### ✅ What you compare (engineer-style)

Compare across the same categories so it’s fair:

* **Goal / problem scope**
* **System setup** (geometry, dataset, hardware, boundary conditions)
* **Method** (experiment? CFD? ML? analytical model?)
* **Metrics** (accuracy? max temperature? MAE? cost? runtime?)
* **Results**
* **Limitations**
* **Best use-case** (when should *you* pick this method?)

### Quick comparison template (what students will fill)

| Category      | Paper A | Paper B | Who wins? / Why |
| ------------- | ------- | ------- | --------------- |
| Method        |         |         |                 |
| Data / Setup  |         |         |                 |
| Key results   |         |         |                 |
| Limitations   |         |         |                 |
| Best use-case |         |         |                 |

***

## Skill C — Extract actionable technical details (not just “ideas”)

When you read for **implementation**, hunt for:

* **Equations & symbol definitions**
* **Boundary conditions / assumptions**
* **Parameter values** (materials, geometry, hyperparameters)
* **Datasets** (source, size, preprocessing)
* **Algorithm steps** (pseudo-code hints)
* **Failure cases** (“when it didn’t work” is gold)

### 🔎 The “Implementation Scanner” checklist

Ask the AI to find:

* ✅ What would I need to replicate this in a lab or simulation?
* ✅ What values do I need?
* ✅ What is missing or unclear?

***

## Skill D — Read & synthesize from forums (Reddit / Stack Exchange / ResearchGate)

Forums are great for:

* “Why does my simulation blow up?”
* “Which assumption is invalid?”
* “What does this error mean?”
* “What’s the practical fix?”

**But** forums contain noise, opinions, and half-truths. So you must:

* Extract the **claim**
* Find the **evidence** (links, equations, references)
* Cross-check with **official docs** or papers

Useful places:

* Engineering Stack Exchange home (browse relevant tags) [\[engineerin...change.com\]](https://engineering.stackexchange.com/)
* Eng-Tips engineering forums (real industry chatter) [\[eng-tips.com\]](https://www.eng-tips.com/)
* ResearchGate (papers + Q/A, but verify carefully) [\[researchgate.net\]](https://www.researchgate.net/profile/Adrian-Calborean/publication/397922801_Phase_Change_Materials_for_Thermal_Management_in_Lithium-Ion_Battery_Packs_A_Review/links/692548a5acf4cf638537d381/Phase-Change-Materials-for-Thermal-Management-in-Lithium-Ion-Battery-Packs-A-Review.pdf)

***

# 3) Live Demonstrations + Step-by-Step Workflows (copy-paste prompts) — 10–65 min

## Demo setup (lecturer-friendly)

### What you need

* Any LLM tool (ChatGPT / Copilot / Claude / etc.)
* At least one PDF (paper)
* A note doc (Word/Google Doc/Notepad) to paste outputs

### 🔥 Rule of the session

**No PDF = weak results.**  
If your tool allows uploads, **upload the PDF**.

***

## Workflow 1 — **Single Paper Summary (technical depth + evidence)**

### Step-by-step (students follow)

1. Upload paper PDF (or paste abstract + key sections)
2. Run “structured summary” prompt
3. Run “evidence check” prompt
4. Export into a short report paragraph

### ✅ Prompt Template 1: “5-Box Technical Summary”

Copy-paste:

```text
You are my engineering reading assistant.

Task: Create a technical summary of this paper using the 5-Box format:
1) Problem & context
2) Approach (methods, setup, assumptions)
3) Key results (include numbers if available)
4) Limitations / failure cases / what is unclear
5) Practical takeaway (how I could apply it in a design or project)

Constraints:
- Use short bullet points.
- If a claim is not directly supported in the paper, label it as “NOT FOUND”.
- Add a “Where in the paper?” line for each box (section/page/table/figure).

Output length: 250–400 words.
```

### ✅ Prompt Template 2: “Evidence-First Summary (anti-hallucination)”

```text
Before summarizing, list the top 10 most important factual claims in the paper.

For each claim:
- Quote the exact supporting sentence (short)
- Identify where it appears (page/section/figure/table)
- Rate confidence: High / Medium / Low

Then write a 200-word summary using ONLY the High/Medium claims.
```

> Why this works: it forces the model to **anchor claims** to the text.

***

## Workflow 2 — **Compare Two Papers Side-by-Side**

### Step-by-step

1. Upload Paper A + Paper B
2. Ask for a comparison table
3. Ask for “who wins for which scenario”
4. Ask for “limitations + missing experiments”

### ✅ Prompt Template 3: “Comparison Table + Verdict”

```text
You will compare Paper A and Paper B.

Output a table with:
- Research goal
- Methodology (experiment/simulation/ML), key assumptions
- Data/setup (geometry, materials, dataset size, boundary conditions)
- Evaluation metrics
- Key results (numbers)
- Limitations and risks
- Best use-case (when I should choose it)

Rules:
- If something is not stated, write “NOT STATED”.
- Do not invent values.
- After the table, give 5 bullet “engineering takeaways”.
```

### ✅ Prompt Template 4: “Method Difference Spotlight”

```text
Focus ONLY on methodology differences.

Answer:
1) What does Paper A do that Paper B does not?
2) What does Paper B do that Paper A does not?
3) Which assumptions differ and why that matters?
4) What result differences could be caused by setup differences (not the idea)?
```

***

## Workflow 3 — **Critical Review / Gap Analysis** (the “Reviewer Brain”)

### ✅ Prompt Template 5: “Peer Reviewer Mode”

```text
Act like a strict but fair peer reviewer.

Write:
- 3 strengths (technical + clarity)
- 3 weaknesses (methods, missing details, validity threats)
- 3 questions you would ask the authors
- 2 experiments or ablation studies that would increase confidence
- 2 risks if someone uses this method in real engineering

Be specific. Refer to sections/figures when possible.
If you can’t find evidence, say “NOT FOUND”.
```

### ✅ Prompt Template 6: “Red Flag Finder”

```text
Scan the paper and list possible red flags:
- unclear assumptions
- missing baseline comparisons
- inconsistent units
- results without error bars/uncertainty
- claims that sound too strong for the evidence

For each red flag, quote the relevant part or say “NOT FOUND”.
```

***

## Workflow 4 — Turn Paper Insights into a **Short Technical Report / Blog Post**

### ✅ Prompt Template 7: “1–2 Page Technical Brief”

```text
Write a 1–2 page technical brief for an engineering student audience.

Topic: (paper topic)

Include:
- Problem statement (why it matters)
- What the paper did (simple explanation)
- Key results (with numbers if available)
- Design implications (how it changes decisions)
- Limitations and when NOT to use it
- 3 references: this paper + 2 related sources mentioned in the paper (if present)

Style:
- Short sections with headings
- Bullets and one small comparison table
- Define jargon in parentheses the first time
```

### ✅ Prompt Template 8: “Blog Post (clear + practical)”

```text
Turn this paper into a blog post titled:
“3 Things Engineers Can Steal from This Paper (Without Copying)”

Constraints:
- No hype. No marketing tone.
- Use simple language.
- Add a “Try this in your project” section with 3 steps.
- Add a “What the paper does NOT prove” section.
```

***

# 4) Critical Warnings & Best Practices (emphasize strongly) — 65–80 min 🚨

## 🚨 WARNING BOX: Avoid the Hallucination Trap

LLMs can sound **super confident** while inventing:

* equations
* API endpoints
* dataset sizes
* experimental conditions
* performance numbers

### ✅ The safe rule

> If it’s not in the PDF / source, it’s **not real**.

***

## “Weak vs Strong Prompt” (show live)

### ❌ Weak prompt

> “Summarize this battery thermal paper.”

What you often get:

* vague summary
* missing metrics
* invented details

### ✅ Strong prompt

> “Summarize using 5-box format + include evidence + mark NOT FOUND.”

What you get:

* structured summary
* traceable claims
* fewer hallucinations

***

## 🧱 Context is King: How to feed the model properly

### Give it:

* The **PDF** (best)
* Or the **abstract + methods + results + conclusion**
* Your goal: “I’m designing X, I need Y”

### Add “project rules” (simple version)

Example “rules” you paste at top of every chat:

```text
Project rules:
- Do not invent details.
- If unsure, say “I’m not sure”.
- Prefer quotes + locations over guesses.
- Use SI units unless paper uses otherwise (then keep original).
- Always list assumptions explicitly.
```

> In coding projects, people often store rules in files like `.cursorrules` or similar to keep behavior consistent across sessions. (Same idea: **consistent constraints**.)

***

## ✅ Verification Ladder (fast, realistic)

Use this whenever output matters (lab report, design, safety):

1. **Does the paper actually say that?** (find the quote)
2. **Do the units make sense?** (K vs °C mistakes happen)
3. **Can I reproduce the key number?** (quick calc / sanity check)
4. **Does another source agree?** (second paper / textbook / datasheet)
5. **What if it’s wrong?** (risk thinking)

***

## 🧰 Tool Strategy: Which tool for which job?

Here’s a simple guide (not sponsored, just practical):

### General chatbots (fast drafting + structure)

* Great for: summarizing, outlining, rewriting
* Risk: may hallucinate if not grounded

### “Source-grounded” research tools (strong for papers)

* **NotebookLM**: works on your uploaded sources and shows citations to those sources, which helps verification. [\[notebooklm.google\]](https://notebooklm.google/?hl=en-GB), [\[workspace.google.com\]](https://workspace.google.com/products/notebooklm/)
* **Perplexity**: “answer engine” style with citations to web sources (good for quick background + cross-checks). [\[perplexity.ai\]](https://www.perplexity.ai/), [\[builtin.com\]](https://builtin.com/artificial-intelligence/what-is-perplexity-ai)

### Writing quality tools (style consistency, compliance)

* **Acrolinx**: focuses on writing standards and content governance (useful for professional documentation teams). [\[acrolinx.com\]](https://www.acrolinx.com/product/), [\[acrolinx.com\]](https://www.acrolinx.com/for-product/)

> Simple takeaway:  
> **For papers:** prefer tools that **cite sources**. [\[notebooklm.google\]](https://notebooklm.google/?hl=en-GB), [\[perplexity.ai\]](https://www.perplexity.ai/)

***

## ✍️ Micro-writing best practice (equations & symbols)

If you include equations in writing:

* Define symbols clearly (“where …”) and format equations so they don’t slow comprehension. [\[chec.engin...ornell.edu\]](https://chec.engineering.cornell.edu/writing-numbers-and-equations/)
* Keep number/unit formatting consistent (small mistakes look unprofessional). [\[chec.engin...ornell.edu\]](https://chec.engineering.cornell.edu/writing-numbers-and-equations/)

***

# 5) Active Learning & Engagement Activities (3–4 interactive elements) — built into the session ✅

## Activity 1 (Live, 5–8 min): **Prompt Makeover**

### Lecturer shows a “bad prompt”

> “Summarize this paper and tell me the key equations.”

### Students improve it together (guiding questions)

Ask students:

* “What output format do you want?”
* “How do we prevent made-up equations?”
* “How do we force evidence?”

### Example upgraded prompt (class builds)

```text
Summarize using 5-box format.
Extract equations ONLY if you can quote them and say where they appear.
If an equation is not clearly shown, write “NOT FOUND”.
```

***

## Activity 2 (Pair work, 15–20 min): **Summarize + Compare Two Real Papers**

Pick a theme (mechanical/energy) and give two open papers.

### Paper set (Battery Thermal Management)

**Paper A:** “Li-Ion Battery Thermal Characterization for Thermal Management Design” (open PDF) [\[docs.nrel.gov\]](https://docs.nrel.gov/docs/fy24osti/89032.pdf)

* Focus: thermal characterization using isothermal calorimetry; emphasizes temperatures/C-rates and module-level effects. [\[docs.nrel.gov\]](https://docs.nrel.gov/docs/fy24osti/89032.pdf)

**Paper B:** “Thermal Management of Lithium-Ion Batteries: PCM vs Air Cooling with Fins” (arXiv PDF) [\[arxiv.org\]](https://arxiv.org/pdf/2503.10244)

* Focus: simulation comparing PCM vs air cooling; discusses temperature reduction and fin effects. [\[arxiv.org\]](https://arxiv.org/pdf/2503.10244)

### Pair instructions

Each pair must produce:

* A **5-box summary** of Paper A
* A **5-box summary** of Paper B
* A **comparison table** (method, setup, results, limitations)
* A final verdict:
  * “If my project is EV pack design vs lab characterization, which paper is more useful and why?”

> Lecturer tip: Encourage students to write “NOT STATED” when details are missing. That’s good scientific behavior ✅

***

## Activity 3 (Group discussion, 8–10 min): **What could go wrong if we trust AI blindly?**

Prompt questions:

* If AI invents an equation in your report, what happens?
* If AI misreads units (K vs °C), what happens?
* If AI claims a dataset exists but it doesn’t, what happens?
* In safety-critical fields (civil, aerospace), what’s the consequence?

**Expected key points**

* Wrong design decisions
* False confidence
* Academic integrity issues
* Safety risks
* Lost time debugging fake details

***

## Activity 4 (Mini-challenge, 10–15 min): **Turn AI Summary into a 1-page Technical Brief**

### Deliverable (students submit)

A one-page brief with:

* Title
* Problem
* Method
* Key results (with numbers if present)
* One small diagram (optional)
* Limitations + “Do not overclaim” section
* Proper citation of the paper(s)

### Bonus constraint (for stronger students)

Add: “What experiment would you run next?”

***

## Reflection (2–3 min): “When do you still read the full paper?”

Quick answers should include:

* When designing something expensive or risky
* When implementing the method
* When writing your own research
* When results depend on assumptions

***

# 6) Closing & Takeaways — 80–90 min 🎁

## ✅ One-page Cheat Sheet (Prompts + Rules)

### **Golden Rules (print this)**

* **No source → no trust.**
* **Require evidence:** quotes + page/section.
* **Mark missing info:** “NOT FOUND / NOT STATED”.
* **Always sanity-check:** units, orders of magnitude, baseline comparisons.
* **AI drafts. You verify.**

***

## 🧾 Best Prompt Templates (copy-paste library)

### 1) 5-Box Summary (fast)

```text
Summarize using 5-box format (problem, approach, results, limits, takeaway).
Use bullets. Include numbers. Add “Where in the paper?” for each box.
If not supported, write “NOT FOUND”.
```

### 2) Comparison Table

```text
Compare Paper A vs Paper B in a table: goal, method, setup, metrics, results, limits, best use-case.
Use “NOT STATED” if missing. No invented values.
```

### 3) Gap Analysis

```text
Act as a reviewer: 3 strengths, 3 weaknesses, 3 questions, 2 missing experiments, 2 real-world risks.
Cite sections/figures when possible.
```

### 4) Implementation Extractor

```text
Extract implementation details:
- equations + symbol definitions
- parameter values
- boundary conditions
- dataset/source details
- algorithm steps
Output a “Replication Checklist”.
Anything missing: “NOT FOUND”.
```

### 5) Brief Builder

```text
Write a 1-page technical brief: problem, method, results, design implications, limits, proper citations.
Simple language. No hype. Add “What it does NOT prove”.
```

***

## 🔁 Recommended Workflow for Future Assignments

**Repeatable “Read → Verify → Write” pipeline:**

1. **Skim manually (5 min):** abstract, figures, conclusion
2. **LLM summary (5 min):** 5-box + evidence
3. **Verification pass (10 min):** check 5 key claims in PDF
4. **Comparison (optional):** add second paper + table
5. **Write output:** brief/report/blog using verified notes

***

## 🧑‍⚖️ Ethical reminder: citation + academic integrity

* If AI helps you write, you still must:
  * **Cite the paper**
  * **Cite any external sources**
  * Follow your course policy on AI use
* Don’t submit AI text you haven’t checked. That’s how mistakes spread.

***

# 📚 Realistic Engineering Paper Examples (ready-to-use)

Use these for class activities or homework.

### Energy / Mechanical (battery thermal management)

* NREL open PDF: “Li-Ion Battery Thermal Characterization for Thermal Management Design” [\[docs.nrel.gov\]](https://docs.nrel.gov/docs/fy24osti/89032.pdf)
* arXiv PDF: “PCM vs Air Cooling Systems Equipped with Fins” [\[arxiv.org\]](https://arxiv.org/pdf/2503.10244)
* arXiv PDF (hybrid microchannels + PCM + nanofluid idea): “Compact Hybrid Battery Thermal Management System…” [\[arxiv.org\]](https://arxiv.org/pdf/2412.00999)

### Civil / Infrastructure (structural health monitoring)

* arXiv: “Deep learning for structural health monitoring…” [\[arxiv.org\]](https://arxiv.org/abs/2211.10351), [\[arxiv.org\]](https://arxiv.org/pdf/2211.10351)
* arXiv: “Foundation Models for Structural Health Monitoring” [\[arxiv.org\]](https://arxiv.org/abs/2404.02944)

> Lecturer tip: Pick **two papers with different methods** (experiment vs simulation vs ML) so the comparison is meaningful.

***

# 🔎 Further Resources (quick, practical)

### Tools (source-grounded reading)

* NotebookLM (grounded in your sources, provides citations) [\[notebooklm.google\]](https://notebooklm.google/?hl=en-GB), [\[workspace.google.com\]](https://workspace.google.com/products/notebooklm/)
* Perplexity (AI answer engine with citations for web research) [\[perplexity.ai\]](https://www.perplexity.ai/), [\[builtin.com\]](https://builtin.com/artificial-intelligence/what-is-perplexity-ai)

### Writing clarity (equations & numbers)

* Cornell Engineering Communication: Numbers & equations in writing [\[chec.engin...ornell.edu\]](https://chec.engineering.cornell.edu/writing-numbers-and-equations/)

### Forums (for real-world troubleshooting)

* Engineering Stack Exchange [\[engineerin...change.com\]](https://engineering.stackexchange.com/)
* Eng-Tips [\[eng-tips.com\]](https://www.eng-tips.com/)

***

# 🎒 Optional Homework (easy to grade)

1. Choose **one paper** → produce a 5-box summary + 5 verified quotes.
2. Choose **two papers** → comparison table + 1-paragraph verdict.
3. Add a “Hallucination Audit”: list 3 claims the AI made that were **NOT FOUND**.

***

## Quick follow-up question (so I can tailor it)

Do you want this tutorial packaged as:

1. **Lecture script + slide bullets**, or
2. **Student worksheet** (fillable templates + grading rubric), or
3. Both?
