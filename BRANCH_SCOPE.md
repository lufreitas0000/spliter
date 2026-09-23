# Branch 4: PyMuPDF Tag/Bookmark Extraction (Phase 2 Enhancement)

## Task
Enhance the router or spatial layer to read native PDF bookmarks, TOCs, and embedded tags using PyMuPDF bindings, outputting them as an intermediate data structure.

## Context
This branch is part of parallel development efforts to avoid merge conflicts.

## Why it's safe
This is a localized enhancement to a specific adapter/infrastructure layer. As long as the interface (ports.py) remains stable, it won't break other teams' work.