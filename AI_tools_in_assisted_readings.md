# Artificial Intelligence Literacy II – AI for Engineers
**Tutorial 1:** Tools in Assisted Readings – Exploring the Limitations of Large Language Models (LLMs)\
**Date:** 13/4/2026\
**Instructor:** JOE

>***[!WARNING]***\
>**CRITICAL SAFETY NOTICE:** LLMs can be dangerous in engineering contexts because of **hallucinations**—generating false or fabricated information with high confidence. In engineering, relying on unverified AI outputs for calculations, material properties, or system designs can lead to catastrophic failures. Always verify critical information against primary sources.

---

## 1. Introduction: Moving from Speed to Safety

Welcome to Tutorial 1. In Lecture 1, we explored how Artificial Intelligence (AI) can accelerate your workflow—summarizing vast amounts of literature, extracting key data points, and translating complex jargon into digestible concepts. We learned *how* to use the tools. 

Today, we focus on a more critical engineering skill: **how to use them safely.** As future engineers, your work will eventually impact the physical world. A misread structural property or a fabricated chemical threshold can cause systems to fail. While Large Language Models (LLMs) like ChatGPT, Claude, and Gemini are powerful, they are fundamentally limited by their architecture. They do not "know" facts; they predict patterns. This tutorial will look under the hood to understand *why* AI lies, how to spot it, and how to rigorously verify AI-assisted reading.

---

## 2. Why AI Confidently Lies? – The Mathematics of LLM Hallucinations Probability Game

To understand why an LLM hallucinates [1], we must first understand how it generates text. LLMs operate via **autoregressive generation**. Simply put, they are playing a highly sophisticated, high-dimensional game of "autocomplete."

### The Core Equation: Next-Token Prediction
When you give an LLM a prompt, it breaks your text down into chunks called **tokens**. It then calculates the probability of what the very next token should be, based on all the previous tokens. Mathematically, the model evaluates the conditional probability:

$$P(x_{t+1} | x_1, \dots, x_t; \theta)$$

Where:
* $x_{t+1}$ is the next token it needs to guess.
* $x_1, \dots, x_t$ is the sequence of preceding tokens (your prompt + everything it has written so far).
* $\theta$ represents the billions of parameters (weights) learned during training.

### The Objective: Probability ≠ Truth
During training, the model uses a **maximum-likelihood objective**. It tries to minimize the difference between its predictions and the actual text in its massive training dataset. It minimizes the negative log-likelihood loss:

$$\mathcal{L}(\theta) = - \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$
=−t=1∑T​logP(xt​∣x<t​;θ)
Notice what is missing from these equations? **A database of truth.** The model maximizes the *likelihood* of a word appearing next based on human linguistic patterns, not factual correctness. If a fabricated equation *looks* statistically similar to real engineering text, the model will confidently generate it.

### Controlling the Game: Temperature, Top-p, and Top-k
When predicting the next token, the model doesn't just output the single most likely word. It outputs a probability distribution (logits, represented by $z_i$) over its entire vocabulary. We modify these probabilities using a parameter called **Temperature ($T$)**:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* If $T \to 0$: The model always picks the most probable token. It becomes repetitive and robotic.
* If $T = 1$: Standard probability.
* If $T > 1$: The model becomes "creative," flattening the probabilities and sometimes choosing unlikely tokens. In engineering, "creativity" often equals "hallucination."

To prevent complete gibberish, models use **Top-k** (only sampling from the $k$ most likely tokens) and **Top-p** (only sampling from tokens whose cumulative probability exceeds $p$). 

**Pseudo-code for the Autoregressive Loop:**
```python
def generate_text(prompt, max_tokens, model, temperature=0.2):
    sequence = tokenize(prompt)
    for _ in range(max_tokens):
        # 1. Model calculates raw scores (logits) for the next token
        logits = model.forward(sequence)
        
        # 2. Temperature scaling and probability conversion
        probs = softmax(logits / temperature)
        
        # 3. Play the probability game: sample the next token
        next_token = sample(probs, top_k=50, top_p=0.9) 
        
        sequence.append(next_token)
        if next_token == EOS_TOKEN: # End of sentence
            break
            
    return detokenize(sequence)
```

**Concrete Example:** If you ask an LLM about a newly published 2025 paper on "AeroGraphene-X", a paper it wasn't trained on, its probability engine will stitch together plausible-sounding materials science terms—inventing authors, properties, and conclusions—because that sequence yields a high $P(x_{t+1})$. 

---

## 3. Real-World Dangers of Hallucinations in Engineering

In an engineering literature review, hallucinated outputs have severe consequences. Studies have shown that LLMs frequently invent citations that look perfect in APA format but point to non-existent journal articles [2]. 

### Case Study 1: The Fabricated Equation
A civil engineering student used an LLM to find the formula for the shear stress of a non-Newtonian fluid under specific thermal gradients. The LLM generated a beautifully formatted KaTeX equation: $\tau = \mu_{eff} \frac{du}{dy} + \alpha \Delta T$. The equation was entirely hallucinated; the $\alpha \Delta T$ term was statistically blended from solid mechanics (thermal expansion) into fluid dynamics. Using this would have ruined the computational simulation.

### Case Study 2: Invented References
A team researching autonomous drone navigation asked an LLM for papers on "Lidar degradation in heavy fog." The model provided three detailed summaries of papers from *IEEE Transactions on Robotics*. The titles sounded perfect. The DOIs looked real. However, the papers did not exist. The LLM had generated "highly probable" text strings that perfectly mimicked academic citations [2].

**The Takeaway:** In safety-critical systems, patent filings, or academic research, you cannot rely on a probability engine to act as a factual database. 

---

## 4. How to Use LLMs Correctly for Technical Reading

To safely leverage AI for your engineering reading, you must adopt a defensive posture. Use the following **Safe Technical Reading Protocol with LLMs**.

### 6-Step Safe Technical Reading Protocol

1.  **Use Retrieval-Augmented Generation (RAG) Tools:** Instead of vanilla ChatGPT, use tools designed for academic reading. **Perplexity**, **Elicit**, **Scite.ai**, or **Claude Projects** ground their answers in actual PDFs or live search results, reducing hallucinations.
2.  **Upload, Don't Ask:** Never ask an LLM "What are the latest papers on X?" Instead, download the specific PDF you want to read, upload it to the LLM, and prompt it to *only* use the provided document.
3.  **Apply Source-Grounding Prompts:** Force the model to cite the exact section of the text it used.
4.  **Use Chain-of-Verification (CoVe):** Ask the LLM to draft a summary, then prompt it to generate a list of verification questions about its own summary, and answer them based *only* on the text.
5.  **Run Self-Consistency Checks:** Ask the LLM the same question 3 times in 3 different ways. If the data points (like a material's tensile strength) change between outputs, it is hallucinating.
6.  **The Human Audit:** Always manually check equations, numbers, and references against the original PDF.

### Safe Prompt Template (Copy-Paste)
When asking an LLM to summarize an engineering paper, use this prompt structure:

> "I have uploaded an engineering research paper. Act as an expert engineering tutor. 
> 1. Summarize the core methodology and findings in 3 bullet points. 
> 2. You MUST strictly rely ONLY on the provided text. If the text does not contain the answer, explicitly state 'The document does not provide this information.'
> 3. For every claim you make, append a citation with the exact section heading and page number from the uploaded PDF."

---

## 5. Hands-on Exercise (30 Minutes)

**Context:** Below is a genuine (shortened) abstract regarding a new structural composite. Following it are three LLM-generated summaries. Your task is to identify the hallucinations using the principles learned today.

**Original Abstract:**
> *“We introduce a novel carbon-fiber reinforced polymer (CFRP-Z) infused with zinc-oxide nanoparticles. Testing under standard atmospheric conditions ($25^\circ\text{C}$, 1 atm) revealed a tensile strength of $3.2\text{ GPa}$, a $15\%$ improvement over standard CFRPs. However, the material exhibits significant brittleness at cryogenic temperatures (below $-50^\circ\text{C}$), making it unsuitable for high-altitude aerospace applications. Future work will explore polymer matrix alterations to mitigate this thermal degradation.”*

**LLM Summary A:**
"CFRP-Z is a new composite material mixed with zinc-oxide. It achieves a massive tensile strength of $3.2\text{ GPa}$, making it ideal for the aerospace industry, specifically for high-altitude satellites and cryogenic environments."

**LLM Summary B:**
"The researchers developed CFRP-Z, which shows a $15\%$ improvement in tensile strength ($3.2\text{ GPa}$) at $25^\circ\text{C}$. The addition of zinc-oxide nanoparticles increases the ductility of the material at $-50^\circ\text{C}$."

**LLM Summary C:**
"CFRP-Z is a carbon-fiber composite infused with zinc-oxide. It has a tensile strength of $3.2\text{ GPa}$ at room temperature ($25^\circ\text{C}$). However, it is highly brittle at cryogenic temperatures ($-50^\circ\text{C}$), which limits its aerospace applications. The researchers suggest using a titanium matrix in future iterations."

**Your Task:**
1. Form groups of two.
2. Identify the specific hallucination in Summaries A, B, and C.
3. Explain *why* the LLM's autoregressive nature might have produced that specific error (e.g., associating "aerospace" positively with "new carbon-fiber").

---

## 6. Key Takeaways & Checklist

* [ ] **LLMs predict, they don't know.** They output tokens based on highest probability $P(x_{t+1})$, not factual accuracy.
* [ ] **Beware the "Temperature" illusion.** Articulate, confident text does not equal correct text.
* [ ] **Numbers and Citations are vulnerable.** The probability game often fabricates equations, specific data values, and academic references [2].
* [ ] **Ground your models.** Always provide the source text (PDF) and command the model to read *only* from that text.
* [ ] **Always verify.** You are the engineer. The liability for the final output rests on you, not the AI.

---

## 7. References

[1] Huang, L., Yu, W., Ma, W., Zhong, Y., Feng, Z., Wang, H., ... & Peng, H. (2023). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. *arXiv preprint arXiv:2311.05232*.

[2] Alkaissi, H., & McFarlane, S. I. (2023). Artificial hallucinations in ChatGPT: implications in scientific writing. *Cureus*, 15(2).