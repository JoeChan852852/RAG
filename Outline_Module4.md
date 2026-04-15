# AILT9018 Artificial Intelligence Literacy II – AI for Engineers
## Module: Fundamentals in NLP, LLM and RAG

### 1. Module Overview
This three-class module is designed to take engineering students on a progressive, hands-on learning journey from basic AI interactions to building advanced multimodal systems. Starting with a gentle, no-code introduction to LLM APIs, students quickly transition into intermediate Python programming to solve real-world AI hallucination problems using Retrieval-Augmented Generation (RAG). The module culminates in an advanced exploration of Vision-Language Models (VLMs) and joint vector spaces, empowering students to build sophisticated, multimodal retrieval pipelines. By the end of this series, students will have transformed from passive AI consumers into capable AI system architects.

---

### 2. Class 1: Introduction to LLM APIs and No-Code Automation
**Level:** Beginner

**Learning Objectives:**
* Understand the basic concept of Application Programming Interfaces (APIs) without complex jargon.
* Successfully navigate the OpenRouter platform to generate and manage API keys.
* Build and execute a functional AI workflow using a zero-code automation tool (n8n).

**Key Topics:**
* **APIs Demystified:** What they are, how they act as bridges between software, and why engineers use them.
* **The OpenRouter Ecosystem:** Accessing a variety of industry-leading LLMs through a single standardized endpoint.
* **Introduction to n8n:** Understanding node-based visual programming and workflow automation.

**Hands-on Activities & Demo:**
* **Live Setup:** Instructor-led walk-through of creating an OpenRouter account and securing an API key.
* **Interactive Demo:** Students will build their first n8n workflow, configuring an HTTP Request node to send a prompt to an LLM and parse the JSON response.

```mermaid
flowchart TD
    A([🔑 Get API Key from OpenRouter]) --> B[⚙️ Configure n8n HTTP Node]
    B --> C[🚀 Trigger Workflow]
    C --> D[🌐 Call OpenRouter API]
    D --> E[🧠 LLM Processes Request]
    E --> F([📥 Receive & View Output in n8n])
    
    style A fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style F fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
```


**How This Class Helps Students:**
This introductory class removes the intimidation factor of working with code by using a visual interface, giving students an immediate "quick win." It establishes a foundational understanding of how software communicates with AI brains, which is essential before writing custom scripts.

---

### 3. Class 2: Building Your Own RAG Chatbot
**Level:** Intermediate

**Learning Objectives:**
* Identify the limitations of raw LLMs, specifically hallucination and lack of domain-specific knowledge.
* Evaluate the pros and cons of "prompt stuffing" versus retrieval-based approaches.
* Implement a foundational Python-based RAG pipeline using vector embeddings.

**Key Topics:**
* **The Context Problem:** Why LLMs fail with private data and the limitations of context windows.
* **RAG Principles:** The architecture of Retrieval-Augmented Generation (Chunking, Embedding, Searching, Generating).
* **Embeddings 101:** Converting text into mathematical vectors for similarity search.

**Hands-on Activities & Demo:**
* **Progressive Python Experiments:** Students will run code that demonstrates three distinct approaches:
    1.  *Raw LLM:* Asking a niche question and observing a hallucinated answer.
    2.  *Prompt Stuffing:* Injecting a massive document into the prompt and observing latency/token limits.
    3.  *Full RAG Pipeline:* Using a localized vector search to extract only relevant chunks to feed the LLM.

```mermaid
flowchart LR
    %% Phase 1 - Raw LLM (leftmost)
    subgraph phase1 [Phase 1: Raw LLM]
        A1[User Query] --> A2((LLM))
        A2 --> A3["❌ Hallucinated<br>Generic Response"]
    end

    %% Phase 2 - Prompt Stuffing (middle)
    subgraph phase2 [Phase 2: Prompt Stuffing]
        B1[Massive Document] --> B2[Appended to Query]
        B2 --> B3((LLM))
        B3 --> B4["⚠️ Context Limit Reached<br>Lost in Middle"]
    end

    %% Phase 3 - RAG Pipeline (rightmost = best)
    subgraph phase3 [Phase 3: RAG Pipeline]
        C1[Document Chunks] --> C2[(Vector Database)]
        C3[User Query] --> C4[Retrieve Top-K Chunks]
        C2 -.-> C4
        C4 --> C5[Contextualized Prompt]
        C3 --> C5
        C5 --> C6((LLM))
        C6 --> C7["✅ Accurate &<br>Grounded Response"]
    end

    %% Explicit progression arrows (forces correct left-to-right order + shows evolution)
    A3 -.->|"Limited by context"| B1
    B4 -.->|"Improved with RAG"| C1

    %% Subgraph styling
    style phase1 fill:#ffebee,stroke:#c62828,stroke-width:2px
    style phase2 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style phase3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

**How This Class Helps Students:**
By experiencing the failures of raw LLMs and prompt stuffing firsthand, students deeply understand *why* RAG is the industry standard for enterprise AI. The step-by-step Python demo provides them with a tangible, reusable codebase they can apply to their own engineering projects.

---

### 4. Class 3: Vision-Language Models (VLMs) and Multimodal Retrieval
**Level:** Advanced

**Learning Objectives:**
* Understand the underlying mechanics of Vision-Language Models (VLMs).
* Explain how CLIP (Contrastive Language-Image Pretraining) maps text and images into a shared vector space.
* Design and execute a multimodal retrieval system using a Vector Database and cosine similarity.

**Key Topics:**
* **Beyond Text:** The rise of VLMs and their applications in engineering (e.g., visual inspection, robotics).
* **Joint Vector Spaces:** How CLIP translates pixels and words into the same mathematical language.
* **Cosine Similarity:** The mathematical core of determining how "close" an image is to a text description.

**Hands-on Activities & Demo:**
* **Automated Labeling Pipeline:** Using an LLM to automatically generate metadata/captions for a dataset of photos.
* **Vector Search Demo:** Storing image metadata in a Vector Database, calculating embeddings, and retrieving the exact correct photo simply by typing a descriptive text query.

```mermaid
graph TD
    subgraph 1. Data Ingestion & Embedding
        I1[🖼️ Image Dataset] --> I2[🤖 VLM Auto-Captioning]
        I2 --> I3[📝 Text Metadata]
        I1 -.-> I4
        I3 --> I4((CLIP Model))
        I4 --> I5[(Vector Database)]
    end

    subgraph 2. Multimodal Retrieval
        Q1[🔍 User Text Query] --> Q2((CLIP Model))
        Q2 --> Q3[📐 Query Vector]
        I5 -.-> Q4
        Q3 --> Q4{Cosine Similarity Math}
        Q4 --> Q5[🎯 Retrieve Highest Match Photo]
    end
    
    style I4 fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style Q2 fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style I5 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

**How This Class Helps Students:**
This advanced session pushes students to the cutting edge of current AI technology, demonstrating that AI is not limited to text chatbots. Understanding multimodal embeddings equips engineering students to build smart systems capable of processing and retrieving complex, real-world visual data.

---

### 5. Overall Student Benefits
This three-part module strategically builds engineering students' confidence, practical skills, and deep conceptual understanding of modern AI architectures. By beginning with accessible, no-code integrations, students avoid initial syntax friction and focus purely on system logic. As they progress into intermediate and advanced Python applications, they acquire highly sought-after industry skills—specifically building RAG pipelines and handling multimodal vector databases. Ultimately, this course bridges the gap between theoretical AI concepts and practical software engineering, empowering students to design custom, data-grounded AI tools for their future careers.