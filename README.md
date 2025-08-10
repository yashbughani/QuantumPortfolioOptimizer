# Agentic AI for Qiskit Portfolio Optimization

This project is an intelligent agent that uses a conversational interface to solve the portfolio optimization problem using real-world financial data and Qiskit's quantum finance algorithms.

## Features

- **Conversational Interface:** Uses a Large Language Model (via Groq and LangGraph) to understand user requests in natural language.
- **Real-World Data:** Connects to the Alpha Vantage API to fetch up-to-date stock market data.
- **Qiskit Engine:** Employs Qiskit's powerful optimization module to calculate the optimal portfolio based on Modern Portfolio Theory.
- **Multiple Solvers:** Provides results from three different optimizers for comprehensive analysis: a classical `NumPyMinimumEigensolver` and two quantum-inspired algorithms, `SamplingVQE` and `QAOA`.
- **Flexible Input:** Supports both user-defined stock lists and a predefined "default" list of 15 popular stocks.

## Setup

Follow these steps to set up the project environment.

**1. Clone the repository (or download the files):**
```bash
git clone [Your-Repo-URL-Here]
cd [your-repo-name]
```
---
**2. Create and activate a virtual environment:**
(It is highly recommended to use Python 3.11 or newer)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
---
**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
---
**4. Set up API Keys:**
You will need two free API keys for this project. Open the main Python script and replace the placeholder text with your keys:
- **Groq API Key:** Get one at [console.groq.com](https://console.groq.com/)
- **Alpha Vantage API Key:** Get one at [www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
## How to Run

Execute the main script from your terminal:

```bash
python main.py
```
The AI will then greet you and ask for your investment plan.
### Sample Interaction

AI: Hello! Please describe your investment plan...  
You: I want to select 2 stocks from GOOG, MSFT, and NVDA with high risk.  
--- FINAL RESULTS ---  
--- Result for NumPyMinimumEigensolver ---  
Optimal: selection ['MSFT', 'NVDA'], value -0.0054  

--- Result for SamplingVQE ---  
Optimal: selection ['MSFT', 'NVDA'], value -0.0054

--- Result for QAOA ---  
Optimal: selection ['MSFT', 'NVDA'], value -0.0054