                             ┌──────────────────────────────────────────────────────────────────┐
                             │                          User Query                              │
                             │       "How does self-attention mechanism help LLMs?"             │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                        Router Agent                              │
                             │        ┌───────────────────────────────────────────┐             │
                             │        │    Task 1: Route to Vectorstore or Web    │             │
                             │        └───────────────────────────────────────────┘             │
                             │       ┌──────────────────────────┬───────────────────────────┐   │
                             │       │    if 'self-attention'   │   else                    │   │
                             │       │      in question:        │                           │   │
                             │       │   Output: "vectorstore"  │ Output: "websearch"       │   │
                             │       └──────────────────────────┴───────────────────────────┘   │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                      Retriever Agent                             │
                             │        ┌───────────────────────────────────────────┐             │
                             │        │     Task 2: Retrieve Information          │             │
                             │        └───────────────────────────────────────────┘             │
                             │       ┌──────────────────────────┬───────────────────────────┐   │
                             │       │    If "vectorstore"      │  If "websearch"           │   │
                             │       │      Use PDFSearchTool   │  Use TavilySearchResults  │   │
                             │       └──────────────────────────┴───────────────────────────┘   │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                   Relevance Grader Agent                         │
                             │        ┌───────────────────────────────────────────┐             │
                             │        │   Task 3: Grade Relevance of Content      │             │
                             │        └───────────────────────────────────────────┘             │
                             │               Output: "yes" if relevant, "no" otherwise          │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                Hallucination Grader Agent                        │
                             │        ┌───────────────────────────────────────────┐             │
                             │        │ Task 4: Check for Hallucination           │             │
                             │        └───────────────────────────────────────────┘             │
                             │             Output: "yes" if factual, "no" otherwise             │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                     Final Answer Grader Agent                    │
                             │        ┌───────────────────────────────────────────┐             │
                             │        │   Task 5: Grade and Generate Answer      │              │
                             │        └───────────────────────────────────────────┘             │
                             │        ┌──────────────────────────────┬───────────────────────┐  │
                             │        │    If "yes":                 │ If "no":              │  │
                             │        │  Generate final answer       │ Web search            │  │
                             │        │  with relevant content       │ and response          │  │
                             │        └──────────────────────────────┴───────────────────────┘  │
                             └──────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │
                                                        ▼
                             ┌──────────────────────────────────────────────────────────────────┐
                             │                        Final Answer                              │
                             │            "Self-attention helps LLMs by..."                     │
                             └──────────────────────────────────────────────────────────────────┘