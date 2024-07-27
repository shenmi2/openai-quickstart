import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# 自定义的prompt
CUSTOM_PROMPT = ("你是一个资深的鲜花销售专家，曾经创造连续520个月当选销售冠军的神话。"
                 "你是一个情商很高的人，当你无法回答问题时，不要说你是个机器人或者直接说你无法回答这个问题，而是用很高情商的话术来夸赞顾客的气质可以驾驭店里的每一朵鲜花，从而引导顾客更加心甘情愿地购买鲜花，"
                 "比如：这个问题很深奥，就像您一样高深莫测有内涵。我相信以您的气质，绝对可以搭配小店的任何一朵花~。"
                 "注意不要让用户感受到你是机器人。")


def initialize_sales_bot(vector_store_dir: str = "real_estates_sale"):
    db = FAISS.load_local(vector_store_dir,
                          OpenAIEmbeddings(api_key="sk-AlhxlLBU0BPYJD6I8870F34f377b47Dd8c1f86Ac73AfBe17",
                                           base_url="https://api.xiaoai.plus/v1"), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(api_key="sk-AlhxlLBU0BPYJD6I8870F34f377b47Dd8c1f86Ac73AfBe17",
                     base_url="https://api.xiaoai.plus/v1", model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})
    )
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": f"{message}"})

    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    elif enable_chat:  # 如果开启了大模型聊天模式，则使用prompt来限制输出
        llm = ChatOpenAI(model_name='gpt-4o', api_key="sk-AlhxlLBU0BPYJD6I8870F34f377b47Dd8c1f86Ac73AfBe17",
                         temperature=1,
                         base_url="https://api.xiaoai.plus/v1")
        messages = [
            ("system", CUSTOM_PROMPT),
            ("human", message)
        ]
        response = llm.invoke(messages)
        print(f"[result]{response.content}")
        print(f"[source_documents]{ans['source_documents']}")
        return response.content
    # 否则输出自定义话术
    else:
        return "这个问题很深奥，就像您一样高深莫测有内涵。我相信以您的气质，绝对可以搭配小店的任何一朵花~"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="鲜花销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化鲜花销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
