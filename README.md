# Financial ML & LLM Analysis Assistant

Financial analysis assistant combining XGBoost stock prediction, Excel processing, LLM explanations, and chat memory.

## Project Overview

A FastAPI web application that combines financial-statement data processing, an XGBoost binary classifier, LLM-generated explanations, Excel upload, and SQLite-backed chat history. The ML target used by the source is “Yükselme Tahmini”.

## Features

- Automatic training from an Excel dataset when model artifacts are missing
- Numeric/categorical preprocessing with scikit-learn
- XGBoost binary classification and probability output
- Excel upload and per-row inference
- OpenRouter-powered financial metric explanations
- SQLite chat-history persistence
- Simple browser-based chat/upload interface

## Tech Stack

- Python
- FastAPI
- pandas
- scikit-learn
- XGBoost
- OpenRouter / OpenAI-compatible client
- SQLite
- HTML/JavaScript

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Environment variables

Create a local `.env` file from `.env.example` and fill in your own credentials/settings. Never commit `.env`.

## Usage

```powershell
python app.py
```

## Notes / Limitations

The training Excel file is not included. This repository should be presented as an ML + LLM analysis prototype, not as an automated trading system or investment recommendation engine.

## Background

This project was developed as part of an AI/LLM training program and has been organized here as a standalone portfolio project. The original exercise prompt was not included in the available source files, so this README describes only what the implementation itself supports.

## License

No license has been added yet.
