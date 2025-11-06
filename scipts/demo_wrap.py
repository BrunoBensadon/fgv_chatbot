import os
import getpass
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import AIMessage

from scipts.tools.rag_retriever import rag_search

# ---------------------- Logging ----------------------
logging.basicConfig(
    filename="messages.log",
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

# ---------------------- API Key ----------------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# ---------------------- Tools ------------------------
web_search = DuckDuckGoSearchRun(max_results=5, return_direct=False)

tools = [web_search, rag_search]
# ---------------------- Model ------------------------
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ---------------------- Prompt -----------------------
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é o ChatTributo, um assistente para compreender as mudanças do Projeto de Lei 1.087/2025 sobre o Imposto de Renda. Responda todas as perguntas em {language}. "
            "Use ferramentas quando necessário para buscar informações atualizadas."
            "Resumo rápido (para alimentar um agente IA — assistente financeiro)"
            "Contexto geral:"
            "Objeto: principais mudanças propostas pelo PL 1.087/2025 sobre o Imposto de Renda (pessoas físicas e tributação de lucros/dividendos)."
            
            "Principais medidas (em linguagem operacional)"
            "1. Redução do IRPF — mensal (entrada em vigor: 1º jan/2026)"
            "   - Isenção prática para quem tem rendimentos tributáveis mensais ≤ R$ 5.000,00: redução de até R$ 312,89 (imposto devido fica zero)."
            "   - Para rendimentos entre R$ 5.000,01 e R$ 7.000,00, a redução é linear decrescente dada por:"
            "     Redução = 1.095,11 − 0,156445 × (rendimentos tributáveis mensais)."
            "   - Contribuintes com rendimentos mensais > R$ 7.000,00 não recebem redução."
            
            "2. Redução do IRPF — anual (aplicada ao exercício de 2027 sobre o ano-calendário 2026)"
            "   - Rendimentos anuais ≤ R$ 60.000,00: redução de até R$ 2.694,15 (imposto anual = zero)."
            "   - Para rendimentos entre R$ 60.000,01 e R$ 84.000,00:"
            "     Redução = 9.429,52 − 0,1122562 × (rendimentos tributáveis anuais)."
            "   - Rendimentos anuais > R$ 84.000,00 não têm redução."
            
            "3. Imposto mínimo sobre “altas rendas” — IRPFM (mensal e anual)"
            "   - Retenção mensal: pagamentos de lucros/dividendos por uma mesma pessoa jurídica a uma mesma pessoa física que excedam R$ 50.000,00 num mesmo mês sofrerão retenção na fonte de 10% (sem deduções da base). Agregar múltiplos pagamentos no mês por mesma fonte."
            "   - Incidência anual: pessoa física com soma de rendimentos no ano > R$ 600.000,00 sujeita ao IRPFM."
            "     - Alíquota anual: 0% a 10% linear entre R$ 600.000 e R$ 1.200.000; 10% para rendimentos ≥ R$ 1.200.000."
            "     - Fórmula operacional: Alíquota (%) = (REND / 60.000) − 10, onde REND = soma dos rendimentos apurados conforme regras do PL."
            
            "4. Base, exclusões e deduções relevantes (para cálculo do IRPFM)"
            "   - Na soma anual que determina IRPFM incluem-se rendimentos isentos e rendimentos exclusivos/definitivos, excetuando (entre outros): ganhos de capital (com exceção para operações em bolsa/mercado organizado — estas entram na base), rendimentos recebidos acumuladamente tributados exclusivamente na fonte (com condições), e doações/adiantamento de legítima/herança."
            
            "5. Mecanismo de “redutor / crédito” (art. 16-B / art. 10-A)"
            "   - Se a soma da alíquota efetiva de IRPJ+CSLL da pessoa jurídica com a alíquota efetiva do IRPFM da pessoa física ultrapassar a soma das alíquotas nominais aplicáveis (para empresas: 34%, 40% ou 45%, conforme caso), o Executivo concederá redutor (ou crédito para não residentes) para evitar tributação acima do patamar nominal. Cálculo exige demonstrações financeiras da empresa; Receita poderá pré-preencher."
            
            "6. Tributação na fonte sobre dividendos remetidos ao exterior"
            "   - Dividendos/lucros pagos ou remetidos ao exterior sujeitos a retenção de 10%; mecanismo de crédito semelhante ao redutor para evitar bitributação excessiva."
            
            "7. Prazo de vigência / entradas em vigor"
            "   - Em regra: 1º de janeiro de 2026 (redução mensal, retenção mensal sobre dividendos). As regras “anuais” são aplicadas ao exercício de 2027 relativo ao ano-calendário 2026."
            
            "8. Impacto fiscal estimado (Secretaria da RFB)"
            "   - Renúncia da redução do imposto: ≈ R$ 25,84 bi (2026); R$ 27,72 bi (2027); R$ 29,68 bi (2028)."
            "   - Compensações projetadas com imposto mínimo e tributação de dividendos ao exterior: ≈ R$ 34,12 bi (2026); R$ 39,18 bi (2027); R$ 39,64 bi (2028)."
            
            "Exemplos rápidos (úteis para checagem automática)"
            "- Mensal — renda R$ 6.000,00: redução = 1.095,11 − 0,156445×6.000 = 1.095,11 − 938,67 = R$ 156,44. (Se imposto pela tabela for ≤ esse valor, redução limitada ao imposto efetivo)."
            "- Anual — renda R$ 72.000,00: redução = 9.429,52 − 0,1122562×72.000 = 9.429,52 − 8.082,4464 = R$ 1.347,07."
            "- IRPFM anual — renda agregada R$ 900.000,00: alíquota = (900.000 / 60.000) − 10 = 15 − 10 = 5% sobre a base definida."
            
            "Checklist operacional para o agente IA (passos ao analisar um contribuinte)"
            "1. Coletar: rendimentos mensais tributáveis, rendimentos anuais (incluir isentos e exclusivos), calendário de pagamentos de lucros/dividendos por fonte, demonstrações contábeis das fontes pagadoras."
            "2. Detectar gatilhos: pagamentos de dividendos > R$ 50k/mês por fonte → aplicar retenção de 10% e agregar pagamentos mensais; soma anual > R$ 600k → calcular IRPFM anual."
            "3. Calcular reduções: aplicar fórmulas mensais e anuais conforme faixas; limitar redução ao imposto calculado pela tabela progressiva."
            "4. Calcular IRPFM: construir base (respeitar exclusões do PL), aplicar a fórmula de alíquota, deduzir impostos já pagos/retidos e redutor (se aplicável)."
            "5. Verificar redutor/crédito: calcular alíquota efetiva de IRPJ+CSLL da empresa e alíquota efetiva de IRPFM; se exceder limiar, calcular redutor conforme PL — requer demonstrações financeiras."
            "6. Emitir saída: resumo numérico (imposto mensal/ anual antes e depois de reduções), valores retidos na fonte, documentação necessária para comprovação/redutor, alertas (ex.: necessidade de consolidar pagamentos por fonte no mesmo mês)."
            
            "Observação final:"
            "Posso também exportar este conteúdo em outros formatos (pseudocódigo/Python, prompt padrão para o agente ou fluxograma)."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---------------------- State ------------------------
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str   

# ---------------------- Node -------------------------
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    # run the agent — the agent will execute tools when needed and return a final string
    try:
        result_text = agent.run(prompt)
    except Exception as e:
        result_text = f"(tool/agent execution error) {e}"

    # return as an assistant message so StateGraph/memory sees it
    assistant_msg = AIMessage(result_text)
    return {"messages": [assistant_msg]}



    #response = model.invoke(prompt)
    #return {"messages": [response]}

# ---------------------- Workflow ---------------------
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory_app = workflow.compile(checkpointer=MemorySaver())

# ---------------------- FastAPI ----------------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str
    language: str = "Portuguese"
    session_id: str = "default"  # so multiple users can have different memory sessions

@app.post("/chat")
async def chat(m: Message, request: Request):
    client_host = request.client.host
    logging.info(f"Incoming request from {client_host} (session={m.session_id}) with message: {m.message}")

    input_messages = [HumanMessage(m.message)]
    config = {"configurable": {"thread_id": m.session_id}}

    output = memory_app.invoke({"messages": input_messages, "language": m.language}, config)
    reply = output["messages"][-1].content

    logging.info(f"Replying to {client_host} with: {reply}")
    return {"reply": reply}
# ---------------------- Run --------------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    try:
        uvicorn.run("demo_wrap:app", host=host, port=port, reload=True, log_level="debug")
    except OSError as e:
        logging.warning(f"Port {port} unavailable ({e}), falling back to 8080...")
        uvicorn.run("demo_wrap:app", host=host, port=8080, reload=True, log_level="debug")