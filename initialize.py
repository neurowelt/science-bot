import bot

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 OPR/93.0.0.0"

chatbot = bot.ChatBot(
    knowledge_base="./src/data/knowledge_base.pkl",
    entity_base="./src/data/entity_base.pkl",
    encoding_base="./src/data/encoding_base_comp.npz",
    user_agent=user_agent
)