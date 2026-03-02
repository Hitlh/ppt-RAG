import streamlit as st
import os
import sys
import asyncio
import shutil
from pathlib import Path

# ==========================================
# 🛠️ 1. 环境自检与导包 (Environment Check)
# ==========================================
# 确保当前目录在 sys.path 中，防止 import 报错
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # 尝试引入后端引擎
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig
except ImportError as e:
    st.error(f"❌ 严重错误：无法导入 RAG-Anything 库！\n请确保 app.py 位于项目根目录下。\n详细报错: {e}")
    st.stop()

# ==========================================
# 🎨 2. 页面配置 (Page Config)
# ==========================================
st.set_page_config(
    page_title="RAG-Anything Pro (Single Doc Edition)", 
    page_icon="📚", 
    layout="wide"
)

# ==========================================
# 🧠 3. 核心引擎服务 (Service Layer)
# ==========================================
class RAGService:
    """
    RAG 服务封装类：
    负责管理 RAGAnything 实例，确保模型常驻内存，
    避免每次操作都重新加载模型。
    """
    def __init__(self, api_key, base_url, working_dir="./rag_storage6"):
        self.api_key = api_key
        self.base_url = base_url
        self.working_dir = working_dir
        self.rag_instance = None
        
        # 自动创建工作目录
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    def get_engine(self):
        """单例模式获取引擎实例 (Singleton Pattern)"""
        if self.rag_instance is not None:
            return self.rag_instance

        # === 1. 配置参数 ===
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru",  # 指定使用 MinerU 解析器
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # === 2. 定义 LLM 调用函数 ===
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        # === 3. 定义 Embedding 函数 ===
        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=self.api_key,
                base_url=self.base_url,
            ),
        )

        # === 4. 定义视觉模型函数 (Vision) ===
        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            elif image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # === 5. 实例化引擎 ===
        self.rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )
        return self.rag_instance

# 异步运行辅助函数 (解决 Streamlit 同步环境调用异步代码的问题)
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ==========================================
# 🖥️ 4. 前端界面逻辑 (UI Logic)
# ==========================================

# 初始化 Session State
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = ""

# --- 侧边栏 ---
with st.sidebar:
    st.title("📚 RAG 单文档问答器")
    st.caption("Single Document Parse + QA")
    st.divider()
    
    # 配置区域
    api_key = st.text_input("API Key", type="password")
    base_url = st.text_input("Base URL", value="https://api.yunwu.ai/v1")
    
    if st.button("🔌 启动/重置引擎"):
        st.session_state.rag_service = RAGService(api_key, base_url)
        st.session_state.doc_indexed = False
        st.session_state.current_doc_name = ""
        st.session_state.messages = []
        with st.spinner("正在加载 MinerU 模型 (首次运行可能较慢)..."):
            st.session_state.rag_service.get_engine()
        st.success("引擎已就绪！(In-Memory)")
    
    st.divider()
    
    # 单文档上传区域
    uploaded_file = st.file_uploader(
        "📄 上传一个文档进行解析",
        type=['pdf', 'txt', 'docx', 'pptx'],
        accept_multiple_files=False
    )

    if uploaded_file and st.session_state.rag_service:
        st.info(f"当前文件：{uploaded_file.name}")

        if st.button("🚀 解析并注入知识库"):
            engine = st.session_state.rag_service.get_engine()

            upload_dir = "./temp_uploads"
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
            os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner("正在解析文档，请稍候..."):
                    print(f"开始解析文档: {file_path}")
                    run_async(engine.process_document_complete(
                        file_path=file_path,
                        output_dir="./output",
                        parse_method="auto"
                    ))
                st.session_state.doc_indexed = True
                st.session_state.current_doc_name = uploaded_file.name
                st.success(f"✅ 解析完成：{uploaded_file.name}，现在可以问答了。")
            except Exception as e:
                st.session_state.doc_indexed = False
                st.error(f"处理失败: {str(e)}")

# --- 主聊天界面 ---
st.subheader("💬 知识库问答")
if st.session_state.current_doc_name:
    st.caption(f"当前知识库文档：{st.session_state.current_doc_name}")

# 回显历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理输入
if prompt := st.chat_input("基于文档内容提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if st.session_state.rag_service and st.session_state.doc_indexed:
        engine = st.session_state.rag_service.get_engine()
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 调用问答接口
                print(f"用户提问: {prompt}")
                response = run_async(engine.aquery(prompt, mode="hybrid"))
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": str(response)})
    elif st.session_state.rag_service and not st.session_state.doc_indexed:
        st.error("请先上传并解析一个文档，再开始问答。")
    else:
        st.error("请先在左侧初始化引擎！")