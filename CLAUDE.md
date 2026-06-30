# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python project managed with Poetry, targeting Python >=3.11. The venv is stored in-project at `.venv/`.

**Purpose:** Building a dataset for hybrid search using Reciprocal Rank Fusion (RRF), likely for Qdrant evaluation or benchmarking.

## Commands

```bash
# Install dependencies
poetry install

# Add a dependency
poetry add <package>

# Run a script
poetry run python <script.py>

# Activate the venv
poetry shell
```
