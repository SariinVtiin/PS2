import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
import random
import numpy as np
import json
import glob
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain.schema import Document
# Import the new PS2Quiz module
from ps2_quiz import create_ps2_quiz


# Configurar API key do Google diretamente
os.environ["GOOGLE_API_KEY"] = "AIzaSyB2TejwAjn7GCrr2deFC-kiHZn4YGwIuCU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Configurar layout da página Streamlit
st.set_page_config(
    page_title="PS2 Games Expert",
    page_icon="🎮",
    layout="wide"
)

# Título e descrição
st.title("🎮 Especialista em Jogos de PlayStation 2")
st.markdown("Pergunte-me qualquer coisa sobre jogos de PS2!")

@st.cache_resource
def load_multiple_data_sources():
    documents = []
    metadata_list = []
    
    # Criar um container escondível para logs de carregamento
    with st.expander("Detalhes do carregamento de dados", expanded=False):
        loading_placeholder = st.empty()
        logs = []
        
        # 1. Carregar CSV principal de jogos
        try:
            games_df = pd.read_csv("ps2_games_database.csv")
            st.session_state["games_df"] = games_df  # Armazenar para o quiz
            st.session_state["total_games"] = len(games_df)
            
            # Converter DataFrame para documentos
            loader = DataFrameLoader(
                games_df, 
                page_content_column="description"
            )
            csv_docs = loader.load()
            documents.extend(csv_docs)
            
            # Extrair metadados para referência rápida
            for _, row in games_df.iterrows():
                metadata_list.append({
                    "title": row.get("title", ""),
                    "developer": row.get("developer", ""),
                    "year": row.get("year", ""),
                    "genre": row.get("genre", ""),
                    "source_type": "game_database"
                })
            logs.append(f"✅ Carregado banco de dados principal com {len(games_df)} jogos")
        except FileNotFoundError:
            logs.append("⚠️ Arquivo ps2_games_database.csv não encontrado. Carregando apenas fontes alternativas.")
            # Criar um DataFrame vazio para evitar erros no quiz
            games_df = pd.DataFrame({
                'title': ['PlayStation 2 Trivia'],
                'developer': ['Sony'],
                'year': [2000],
                'genre': ['Console'],
                'description': ['O PlayStation 2 é o console mais vendido de todos os tempos.']
            })
            st.session_state["games_df"] = games_df
            
        # 2. Carregar CSVs adicionais (se houver)
        additional_csvs = glob.glob("data/*.csv")
        for csv_file in additional_csvs:
            try:
                loader = CSVLoader(
                    file_path=csv_file,
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                    }
                )
                csv_docs = loader.load()
                documents.extend(csv_docs)
                
                # Adicionar metadados sobre a fonte
                source_name = os.path.basename(csv_file).replace(".csv", "")
                for doc in csv_docs:
                    doc.metadata["source_type"] = f"additional_csv_{source_name}"
                logs.append(f"✅ Carregado CSV adicional: {csv_file}")
            except Exception as e:
                logs.append(f"⚠️ Erro ao carregar CSV adicional {csv_file}: {str(e)}")
        
        # 3. Carregar documentos PDF (artigos, guias, etc.)
        pdf_files = glob.glob("data/*.pdf")
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                pdf_docs = loader.load()
                
                # Adicionar informações sobre a fonte aos metadados
                source_name = os.path.basename(pdf_file).replace(".pdf", "")
                for doc in pdf_docs:
                    doc.metadata["source_type"] = f"pdf_{source_name}"
                
                documents.extend(pdf_docs)
                logs.append(f"✅ Carregado PDF: {pdf_file}")
            except Exception as e:
                logs.append(f"⚠️ Erro ao carregar PDF {pdf_file}: {str(e)}")
        
        # 4. Carregar arquivos de texto (wikipedias, fóruns arquivados, etc.)
        txt_files = glob.glob("data/*.txt")
        for txt_file in txt_files:
            try:
                loader = TextLoader(txt_file)
                txt_docs = loader.load()
                
                # Adicionar informações sobre a fonte aos metadados
                source_name = os.path.basename(txt_file).replace(".txt", "")
                for doc in txt_docs:
                    doc.metadata["source_type"] = f"text_{source_name}"
                    
                documents.extend(txt_docs)
                logs.append(f"✅ Carregado arquivo de texto: {txt_file}")
            except Exception as e:
                logs.append(f"⚠️ Erro ao carregar texto {txt_file}: {str(e)}")
        
        # 5. Carregar dados JSON personalizados (informações adicionais, curiosidades)
        json_files = glob.glob("data/*.json")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Processar JSON para documentos
                for item in json_data:
                    if isinstance(item, dict):
                        # Extrair texto principal e metadados
                        content = item.get("content", "")
                        metadata = {k: v for k, v in item.items() if k != "content"}
                        metadata["source_type"] = f"json_{os.path.basename(json_file).replace('.json', '')}"
                        
                        # Criar documento
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                
                logs.append(f"✅ Carregado JSON: {json_file}")
            except Exception as e:
                logs.append(f"⚠️ Erro ao carregar JSON {json_file}: {str(e)}")
        
        # Mostrar todos os logs em uma única atualização
        loading_placeholder.markdown("\n".join(logs))
    
    # Dividir documentos longos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)
    
    st.success(f"Carregados {len(split_documents)} documentos de {len(set([doc.metadata.get('source_type', '') for doc in split_documents]))} fontes diferentes")
    
    return split_documents, metadata_list

# Initialize the quiz module
def initialize_quiz_module():
    if "quiz_instance" not in st.session_state and "games_df" in st.session_state:
        st.session_state["quiz_instance"] = create_ps2_quiz(
            st.session_state["games_df"], 
            max_questions=5
        )

# Function to go back to chat mode
def back_to_chat():
    st.session_state["quiz_mode"] = False
    st.session_state["quiz_started"] = False
    st.session_state["show_record_form"] = False
    # Não reiniciamos o quiz aqui, apenas voltamos ao modo chat
    # O estado é mantido para evitar duplicação de pontuações
        
# Função para mostrar os recordes na sidebar
def show_records_in_sidebar():
    if "records" in st.session_state and st.session_state["records"]:
        with st.sidebar.expander("🏆 Hall da Fama - Top 3", expanded=False):
            for i, record in enumerate(st.session_state["records"][:3]):
                medals = ["🥇", "🥈", "🥉"]
                medal = medals[i] if i < 3 else "  "
                st.write(f"{medal} **{record['player_name']}** - {record['score']}/{record['max_score']} ({record['percentage']:.1f}%)")
            
            if len(st.session_state["records"]) > 3:
                st.write(f"*...e mais {len(st.session_state['records']) - 3} jogadores*")

# Função para inicializar o modelo e o banco de dados
@st.cache_resource
def initialize_qa_system():
    try:
        # Carregar múltiplas fontes de dados
        documents, metadata_list = load_multiple_data_sources()
        
        # Configurar embeddings e base de conhecimento
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Configurar safety_settings
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        # Configurar LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-001",  # Atualizado para um modelo mais recente
            temperature=0.3,
            top_k=40,
            top_p=0.95,
            safety_settings=safety_settings
        )
        
        # Template melhorado para incluir diferentes fontes
        from langchain.prompts import PromptTemplate
        
        template = """
        Você é um especialista em jogos de PlayStation 2 (PS2). Sua tarefa é responder perguntas sobre jogos de PS2 
        com base nas diversas fontes de informação fornecidas nos documentos e seu conhecimento geral.
        
        Contexto sobre jogos de PS2 (de múltiplas fontes):
        {context}
        
        Pergunta: {question}
        
        Ao responder:
        1. Seja detalhado, informativo e amigável
        2. Diferencie claramente entre informações provenientes da base de dados de jogos, documentos PDF, arquivos de texto e outras fontes
        3. Se a pergunta não estiver relacionada a jogos de PS2, responda apenas sobre jogos de PS2
        4. Se não tiver informações suficientes nos documentos, use seu conhecimento geral sobre jogos de PS2
        5. Mencione explicitamente o tipo de fonte de onde a informação foi extraída (ex: "Segundo nossa base de dados de jogos...", "De acordo com artigos especializados...", etc.)
        
        Sua resposta:
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 7}), # Aumentado para pegar mais contexto
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {str(e)}")
        raise e

# Sidebar com informações
# Sidebar com informações
with st.sidebar:
    st.header("Sobre o Chatbot Aprimorado")
    st.info("""
    Este chatbot é especializado em jogos de PlayStation 2 e utiliza múltiplas fontes de dados:
    - Base de dados de jogos (CSV)
    - Artigos e guias (PDFs)
    - Textos informativos (TXT)
    - Dados estruturados adicionais (JSON)
    
    Faça perguntas sobre qualquer aspecto dos jogos de PS2!
    """)
    
    st.header("Como funciona")
    st.write("""
    As respostas são geradas combinando:
    1. Múltiplas fontes de dados integradas
    2. IA generativa (Google Gemini)
    3. Técnicas avançadas de RAG (Retrieval-Augmented Generation)
    """)
    
    # Instruções para adicionar mais fontes
    st.header("📁 Adicionar Fontes")
    st.write("""
    Para adicionar novas fontes de dados:
    1. Crie uma pasta 'data' no diretório do projeto
    2. Adicione arquivos nos formatos suportados:
       - CSV: dados estruturados
       - PDF: guias, artigos, manuais
       - TXT: textos informativos
       - JSON: dados personalizados
    """)
    
    # Botão de Quiz na sidebar com nome de usuário
    st.header("📝 Modo Quiz")
    
    # Campo para nome do jogador
    player_name = st.text_input("Seu nome para o Quiz:", value=st.session_state.get("player_name", ""))
    
    # Botão para iniciar o quiz
    if st.button("Iniciar Quiz de PS2"):
        # Salvar o nome do jogador na sessão
        st.session_state["player_name"] = player_name
        st.session_state["quiz_mode"] = True
        st.session_state["quiz_started"] = True
        st.session_state["show_record_form"] = False
    
    # Mostrar recordes na sidebar
    if "quiz_instance" in st.session_state:
        show_records_in_sidebar()

# Verificar se estamos no modo quiz
if "quiz_mode" not in st.session_state:
    st.session_state["quiz_mode"] = False

# Inicializar o sistema
qa_chain = initialize_qa_system()

# Mostrar Quiz ou Chat com base no modo atual
# Verificar se estamos no modo quiz
if "quiz_mode" not in st.session_state:
    st.session_state["quiz_mode"] = False
if "last_saved_score" not in st.session_state:
    st.session_state["last_saved_score"] = None

# Inicializar o sistema
qa_chain = initialize_qa_system()

# Mostrar Quiz ou Chat com base no modo atual
if st.session_state.get("quiz_mode", False):
    # Adicionar diagnóstico para depurar o problema, caso necessário
    debug_mode = False  # Definir como True para mostrar informações de diagnóstico
    
    if debug_mode:
        st.write("### Diagnóstico:")
        if "games_df" not in st.session_state:
            st.error("O DataFrame de jogos não foi carregado corretamente na sessão.")
        else:
            st.success(f"DataFrame de jogos carregado com {len(st.session_state['games_df'])} jogos.")
        
        # Mostrar último score salvo
        if "last_saved_score" in st.session_state and st.session_state["last_saved_score"]:
            st.write("Último score salvo:", st.session_state["last_saved_score"])
    
    # Tentar inicializar o quiz
    try:
        initialize_quiz_module()
        if "quiz_instance" in st.session_state:
            st.session_state["quiz_instance"].render_quiz()
        else:
            st.error("Não foi possível criar instância do quiz.")
            if st.button("Voltar ao Chat"):
                back_to_chat()
    except Exception as e:
        st.error(f"Erro ao inicializar o quiz: {str(e)}")
        if debug_mode:
            st.write("Detalhes técnicos do erro:")
            st.code(str(e))
        if st.button("Voltar ao Chat"):
            back_to_chat()
else:
    # Histórico de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Olá! Sou especialista em jogos de PlayStation 2 com acesso a múltiplas fontes de dados. Como posso ajudar você hoje?"}
        ]

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta sobre jogos de PS2..."):
        # Adicionar pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Exibir pergunta
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Gerar resposta
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando...")
            
            try:
                # Processar pergunta
                result = qa_chain({"query": prompt})
                response = result["result"]
                sources = result["source_documents"]
                
                # Adicionar fontes à resposta com informações sobre tipo de fonte
                if sources:
                    response_with_sources = response + "\n\n**Fontes:**"
                    sources_dict = {}  # Para agrupar por tipo de fonte
                    
                    for doc in sources[:5]:  # Aumentado para mostrar mais fontes
                        source_type = doc.metadata.get("source_type", "Desconhecido")
                        
                        # Criar entrada para o tipo de fonte se ainda não existir
                        if source_type not in sources_dict:
                            sources_dict[source_type] = []
                        
                        # Adicionar informações relevantes com base no tipo de fonte
                        if source_type == "game_database":
                            title = doc.metadata.get("title", "Desconhecido")
                            year = doc.metadata.get("year", "")
                            developer = doc.metadata.get("developer", "")
                            source_info = f"- {title} ({year}, {developer})"
                            if source_info not in sources_dict[source_type]:
                                sources_dict[source_type].append(source_info)
                        elif source_type.startswith("pdf_"):
                            # Para PDFs, mostrar página e nome do arquivo
                            pdf_name = source_type.replace("pdf_", "")
                            page = doc.metadata.get("page", "")
                            source_info = f"- {pdf_name} (Pág. {page})"
                            if source_info not in sources_dict[source_type]:
                                sources_dict[source_type].append(source_info)
                        elif source_type.startswith("text_"):
                            # Para arquivos de texto, mostrar o nome
                            text_name = source_type.replace("text_", "")
                            source_info = f"- Documento: {text_name}"
                            if source_info not in sources_dict[source_type]:
                                sources_dict[source_type].append(source_info)
                        elif source_type.startswith("json_"):
                            # Para JSON, mostrar título ou nome de coleção
                            json_name = source_type.replace("json_", "")
                            title = doc.metadata.get("title", "")
                            source_info = f"- {json_name}: {title}" if title else f"- Coleção: {json_name}"
                            if source_info not in sources_dict[source_type]:
                                sources_dict[source_type].append(source_info)
                        else:
                            # Para outros tipos de fonte
                            source_info = f"- {source_type}"
                            if source_info not in sources_dict[source_type]:
                                sources_dict[source_type].append(source_info)
                    
                    # Adicionar fontes agrupadas
                    for source_type, source_list in sources_dict.items():
                        # Formatar o tipo de fonte para exibição
                        if source_type == "game_database":
                            display_type = "Base de dados de jogos"
                        elif source_type.startswith("pdf_"):
                            display_type = "Artigo PDF"
                        elif source_type.startswith("text_"):
                            display_type = "Documento de texto"
                        elif source_type.startswith("json_"):
                            display_type = "Dados estruturados"
                        elif source_type.startswith("additional_csv_"):
                            display_type = f"Base de dados: {source_type.replace('additional_csv_', '')}"
                        else:
                            display_type = source_type
                            
                        response_with_sources += f"\n**{display_type}:**"
                        for source in source_list:
                            response_with_sources += f"\n{source}"
                    
                    message_placeholder.markdown(response_with_sources)
                    st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
                else:
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                error_message = f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Footer com informações adicionais
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Desenvolvido com Streamlit, LangChain e Google Gemini")
with col2:
    if "total_games" in st.session_state:
        st.markdown(f"Base de conhecimento: {st.session_state['total_games']} jogos + fontes adicionais")