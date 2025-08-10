import os
import pandas as pd
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from alpha_vantage.timeseries import TimeSeries

os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
os.environ["ALPHA_VANTAGE_API_KEY"] = "ALPHA_VANTAGE_API_KEY"

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",  # Tech
    "JPM", "BAC", "WFC",                      # Finance
    "JNJ", "PFE",                             # Healthcare
    "WMT", "COST",                            # Consumer
    "XOM", "CVX",                             # Energy
    "DIS"                                     # Entertainment
]

def run_qiskit_optimization(parameters: "PortfolioParameters") -> str:
    """
    Takes the final parameters, runs the Qiskit optimization with three different
    solvers using free, real-world data from Alpha Vantage.
    """
    print("\nNODE: RUNNING QISKIT OPTIMIZATION")

    tickers = parameters.tickers
    num_assets = len(tickers)
    budget = parameters.budget
    q = parameters.risk_factor

    try:
        print(f"Fetching data for: {tickers}")
        ts = TimeSeries(key=os.environ["ALPHA_VANTAGE_API_KEY"], output_format='pandas')

        all_data = {}
        for ticker in tickers:
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            all_data[ticker] = data['4. close']

        df = pd.DataFrame(all_data)
        df = df.sort_index(ascending=True)

        daily_returns = df.pct_change().dropna()
        mu = daily_returns.mean().to_numpy()
        sigma = daily_returns.cov().to_numpy()
        print("Successfully fetched and processed real-world data.")

    except Exception as e:
        return f"An error occurred while fetching data from Alpha Vantage: {e}. Please ensure your API key is correct and your tickers are valid."

    portfolio = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()

    def get_result_string(result, solver_name):
        selection_indices = result.x
        selected_tickers = [tickers[i] for i, x in enumerate(selection_indices) if x == 1.0]
        value = result.fval
        report = []
        report.append(f"{solver_name}")
        report.append(f"Optimal selection: {selected_tickers}, value {value:.4f}")
        return "\n".join(report)

    final_report = []

    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    result1 = exact_eigensolver.solve(qp)
    final_report.append(get_result_string(result1, "NumPyMinimumEigensolver"))

    algorithm_globals.random_seed = 1234
    cobyla_svqe = COBYLA(maxiter=500)
    ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla_svqe)
    svqe = MinimumEigenOptimizer(svqe_mes)
    result2 = svqe.solve(qp)
    final_report.append(get_result_string(result2, "SamplingVQE"))

    algorithm_globals.random_seed = 1234
    cobyla_qaoa = COBYLA(maxiter=250)
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla_qaoa, reps=3)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result3 = qaoa.solve(qp)
    final_report.append(get_result_string(result3, "QAOA"))

    return "\n\n".join(final_report)


class PortfolioParameters(BaseModel):
    tickers: Optional[List[str]] = Field(None, description="A list of stock ticker symbols, e.g., ['AAPL', 'GOOG'].")
    budget: Optional[int] = Field(None, description="The number of assets to select from the list.")
    risk_factor: Optional[float] = Field(None, description="The user's risk tolerance (low=0.2, medium=0.5, high=0.8).")


class GraphState(TypedDict):
    messages: List[BaseMessage]
    parameters: PortfolioParameters


llm = ChatGroq(model_name="gemma2-9b-it", temperature=0)
structured_llm = llm.with_structured_output(PortfolioParameters)


def get_missing_parameters(params: PortfolioParameters) -> List[str]:
    missing = []
    if not params.tickers: missing.append("'list of stock tickers'")
    if params.budget is None: missing.append("'budget' (how many assets to select)")
    if params.risk_factor is None: missing.append("'risk tolerance' (low, medium, or high)")
    return missing


def extractor_node(state: GraphState) -> dict:
    print("NODE: EXTRACTOR")
    current_params_dict = state['parameters'].model_dump()
    user_message = state['messages'][-1].content.lower()

    if any(keyword in user_message for keyword in ["default", "standard", "predefined"]):
        print("Default stock list requested.")
        current_params_dict['tickers'] = DEFAULT_STOCKS

    current_params_obj = PortfolioParameters(**current_params_dict)

    prompt = (
        f"You are a precise data extraction tool. The user wants to set up a portfolio. "
        f"The following parameters are still missing: {get_missing_parameters(current_params_obj)}. "
        f"Analyze the last user message to extract them. 'tickers' should be a list of strings. Map risk words 'low' to 0.2, 'medium' to 0.5, and 'high' to 0.8. "
        f"If a value is not present, leave it as null.\n\nLast user message: '{state['messages'][-1].content}'"
    )
    extracted_params = structured_llm.invoke(prompt)

    for key, value in extracted_params.model_dump().items():
        if value is not None: current_params_dict[key] = value
    updated_params = PortfolioParameters(**current_params_dict)
    return {"parameters": updated_params}


def question_asker_node(state: GraphState) -> dict:
    print("NODE: QUESTION ASKER")
    missing_params = get_missing_parameters(state['parameters'])
    if not missing_params: return {}
    ai_message_content = f"I've got some of the details, but I still need to know your {', '.join(missing_params)}. Can you please provide that?"
    ai_message = AIMessage(content=ai_message_content)
    return {"messages": state['messages'] + [ai_message]}


def router(state: GraphState) -> str:
    missing_params = get_missing_parameters(state['parameters'])
    if not missing_params:
        print("ROUTER: All parameters collected. Finishing extraction.")
        return "end"
    else:
        print(f"ROUTER: Missing parameters {missing_params}. Asking question.")
        return "ask_question"


workflow = StateGraph(GraphState)
workflow.add_node("extractor", extractor_node)
workflow.add_node("ask_question", question_asker_node)
workflow.set_entry_point("extractor")
workflow.add_conditional_edges("extractor", router, {"ask_question": "ask_question", "end": END})
workflow.add_edge("ask_question", END)
extraction_app = workflow.compile()

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("ALPHA_VANTAGE_API_KEY"):
        print("\nERROR: Please set both your Groq and Alpha Vantage API keys in the script before running.")
    else:
        state = {"messages": [], "parameters": PortfolioParameters()}
        print("AI: Hello! To optimize your portfolio, I need three things:")
        print("   1. A list of stock tickers (e.g., ['AAPL', 'GOOG']) OR you can ask to use the 'default' list.")
        print("   2. Your budget (how many stocks to select from the list).")
        print("   3. Your risk tolerance (low, medium, or high).")

        while True:
            user_input = input("You: ")
            state["messages"] = state.get("messages", []) + [HumanMessage(content=user_input)]
            result = extraction_app.invoke(state)

            state["parameters"] = result["parameters"]
            state["messages"] = result.get("messages", [])

            if not get_missing_parameters(state["parameters"]):
                print("\nAI: Great! I have all the information.")
                break
            else:
                print(f"\nAI: {state['messages'][-1].content}")

        final_parameters = state["parameters"]
        if final_parameters.budget > len(final_parameters.tickers):
            print("\nERROR:")
            print(
                f"Your budget ({final_parameters.budget}) cannot be greater than the number of available stocks ({len(final_parameters.tickers)}).")
        else:
            optimization_result = run_qiskit_optimization(final_parameters)
            print("\nFINAL RESULTS")
            print(optimization_result)