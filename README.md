# Semantic PDF Pipeline

## 1. Executive Summary
A microservice-oriented monorepo designed to ingest both tractable (structured) and intractable (unstructured/raster) PDFs, and efficiently reduce them into organized, semantically pure Markdown files. The system splits the PDFs into small tractable elements (text, equations, tables, figures) using smart routing, prioritizing deterministic native PDF parsing over expensive OCR fallbacks.

For a detailed phased plan of the implementation, please see the [ROADMAP.md](ROADMAP.md).

## 2. Architecture (Directed Acyclic Graph)
The system is decoupled into three core Python applications, ensuring strict separation of concerns between structure routing, spatial extraction, and tensor-based machine learning.

* **`app_structurizer`:** The topological router and extractor. It analyzes the PDF's internal structure using heuristics (like Shannon Entropy). For digitally structured pages, it extracts native text, equations, and bounding boxes. For raster images, it crops the tensors and delegates them. It serves as the CLI entry point.
* **`app_spatial_compiler`:** Ingests spatial data and text bounding boxes from structured PDFs. Applies Euclidean heuristics to reconstruct exact Markdown hierarchies (headers, paragraphs, math blocks) natively without ML inference.
* **`app_vision_encoder`:** The semantic vision engine. Invoked only when necessary (e.g., for scanned images or embedded figures), it processes image tensors using Vision-Language Models to generate semantic Markdown or ALT text for injection.

## 3. Engineering Codex
* **Bounded Contexts:** Each `app_*` directory must maintain its own `requirements.in` and test suite (`pytest`). No cross-app imports are permitted unless mediated by explicit data contracts.
* **TDD & Immutable Domain:** Adheres strictly to Test-Driven Development (TDD) and SOLID principles, using immutable data classes.
* **Intermediate Representation:** Markdown is the mandatory data transfer protocol between components.
EOF
