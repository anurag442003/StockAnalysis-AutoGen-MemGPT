import os
from dotenv import load_dotenv
import requests
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config
import autogen

load_dotenv()

def get_stock_data(symbol):
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1month&outputsize=12&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

def analyze_market():
    indices = ["SPY", "QQQ", "DIA", "IWM"]
    market_data = {}
    for index in indices:
        market_data[index] = get_stock_data(index)
    return market_data

def create_memgpt_agent(system_message):
    # LM Studio configuration
    config_list = [
        {   
            "model": "NULL",
            "api_base": "http://localhost:5001/v1",
            "api_key": "NULL",
        }
    ]
    config_list_memgpt = [
        {
            "model": None,
            "model_wrapper": "chatml",
            "model_endpoint_type": "webui",
            "context_window": 8192,
            "preset": "memgpt_chat",
            "model_endpoint": "http://localhost:5000",
        }
    ]
    llm_config = {"config_list": config_list, "seed": 42}
    llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

    return create_memgpt_autogen_agent_from_config(
        name="MemGPT",
        llm_config=llm_config_memgpt,
        system_message=system_message,
        interface_kwargs={"show_inner_thoughts": True},
    )

def stock_analysis_conversation():
    market_data = analyze_market()
    
    optimistic_agent = create_memgpt_agent(
        name="Optimistic_Analyst",
        system_message="You are an optimistic stock market analyst. Always look for potential growth and positive aspects in companies and market trends."
    )
    
    pessimistic_agent = create_memgpt_agent(
        name="Pessimistic_Analyst",
        system_message="You are a pessimistic stock market analyst. Always be cautious and look for potential risks and downsides in companies and market trends."
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="A user seeking stock investment advice.",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
    )

    groupchat = autogen.GroupChat(
        agents=[user_proxy, optimistic_agent, pessimistic_agent],
        messages=[],
        max_round=5,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat)

    stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Fetch data for all stocks
    stock_data = {}
    for symbol in stocks:
        data = get_stock_data(symbol)
        if data:
            stock_data[symbol] = data

    message = "Analyze the following stock data for multiple companies and decide which is the best investment option:\n\n"
    for symbol, data in stock_data.items():
        message += f"{symbol} Data:\n{data}\n\n"
    message += "Pessimistic_analyst should provide potential risks for each stock, and Optimistic_analyst should highlight growth opportunities. Then, come to a conclusion about which stock is the best investment option and why."


    user_proxy.initiate_chat(
        manager,
        message=message,
    )

if __name__ == "__main__":
    stock_analysis_conversation()