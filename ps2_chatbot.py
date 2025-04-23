
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
    except FileNotFoundError:
        st.warning("Arquivo ps2_games_database.csv não encontrado. Carregando apenas fontes alternativas.")
    
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
                
            st.info(f"Carregado CSV adicional: {csv_file}")
        except Exception as e:
            st.warning(f"Erro ao carregar CSV adicional {csv_file}: {str(e)}")
    
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
            st.info(f"Carregado PDF: {pdf_file}")
        except Exception as e:
            st.warning(f"Erro ao carregar PDF {pdf_file}: {str(e)}")
    
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
            st.info(f"Carregado arquivo de texto: {txt_file}")
        except Exception as e:
            st.warning(f"Erro ao carregar texto {txt_file}: {str(e)}")
    
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
            
            st.info(f"Carregado JSON: {json_file}")
        except Exception as e:
            st.warning(f"Erro ao carregar JSON {json_file}: {str(e)}")
    
    # Dividir documentos longos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(documents)
    
    st.success(f"Carregados {len(split_documents)} documentos de {len(set([doc.metadata.get('source_type', '') for doc in split_documents]))} fontes diferentes")
    
    return split_documents, metadata_list

# Função para inicializar o modo Quiz
def initialize_quiz():
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "quiz_question_count" not in st.session_state:
        st.session_state["quiz_question_count"] = 0
    if "max_questions" not in st.session_state:
        st.session_state["max_questions"] = 5
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = {}
    if "answered" not in st.session_state:
        st.session_state["answered"] = False
    if "selected_option" not in st.session_state:
        st.session_state["selected_option"] = None

# Função para gerar uma pergunta de quiz
def generate_quiz_question(df):
    # Seleciona um jogo aleatório
    random_game = df.sample(1).iloc[0]
    
    # Decide que tipo de pergunta fazer
    question_types = ["developer", "year", "genre", "character", "trivia"]
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # Pesos para cada tipo de pergunta
    question_type = random.choices(question_types, weights=weights, k=1)[0]
    
    if question_type == "developer":
        question = f"Quem desenvolveu o jogo '{random_game['title']}'?"
        correct_answer = random_game['developer']
        # Gera opções incorretas
        wrong_options = list(df['developer'].sample(3).values)
        
    elif question_type == "year":
        question = f"Em que ano foi lançado '{random_game['title']}'?"
        correct_answer = str(random_game['year'])
        # Anos próximos como opções incorretas
        year_range = range(random_game['year'] - 3, random_game['year'] + 4)
        wrong_options = [str(year) for year in year_range if year != random_game['year']]
        wrong_options = random.sample(wrong_options, 3)
        
    elif question_type == "genre":
        question = f"Qual é o gênero de '{random_game['title']}'?"
        correct_answer = random_game['genre']
        wrong_options = list(df['genre'].sample(3).values)
        
    elif question_type == "character":
        if pd.notna(random_game['characters']) and len(str(random_game['characters'])) > 3:
            question = f"Qual destes personagens NÃO aparece em '{random_game['title']}'?"
            
            # Pega personagens do jogo atual
            characters = str(random_game['characters']).split(", ")
            
            # Busca personagens de outros jogos
            other_characters = []
            for _, game in df.sample(5).iterrows():
                if game['title'] != random_game['title'] and pd.notna(game['characters']):
                    other_chars = str(game['characters']).split(", ")
                    if other_chars:
                        other_characters.extend(other_chars)
            
            if other_characters:
                correct_answer = random.choice(other_characters)  # Personagem que NÃO está no jogo
                # Pega alguns personagens do jogo como opções incorretas
                if len(characters) >= 3:
                    wrong_options = random.sample(characters, 3)
                else:
                    wrong_options = characters + random.sample(other_characters, 3 - len(characters))
            else:
                # Fallback se não encontrar outros personagens
                return generate_quiz_question(df)
        else:
            # Se não tiver personagens, gera outro tipo de pergunta
            return generate_quiz_question(df)
            
    elif question_type == "trivia":
        # Perguntas sobre curiosidades gerais de PS2
        ps2_trivia = [
            {
                "question": "Qual é o console mais vendido de todos os tempos?",
                "correct": "PlayStation 2",
                "options": ["PlayStation 2", "Nintendo DS", "PlayStation 4", "Nintendo Wii"]
            },
            {
                "question": "Quantas unidades de PlayStation 2 foram vendidas em todo o mundo?",
                "correct": "Mais de 155 milhões",
                "options": ["Mais de 155 milhões", "Cerca de 100 milhões", "Menos de 80 milhões", "Exatamente 120 milhões"]
            },
            {
                "question": "Em que ano o PlayStation 2 foi lançado no Japão?",
                "correct": "2000",
                "options": ["2000", "1999", "2001", "2002"]
            },
            {
                "question": "Qual dessas cores NÃO foi um modelo oficial do PlayStation 2?",
                "correct": "Verde Neon",
                "options": ["Verde Neon", "Branco", "Prata", "Azul Aqua"]
            },
            {
                "question": "Qual destes jogos vendeu mais unidades no PlayStation 2?",
                "correct": "Grand Theft Auto: San Andreas",
                "options": ["Grand Theft Auto: San Andreas", "Final Fantasy X", "Metal Gear Solid 2", "God of War"]
            }
        ]
        
        trivia_q = random.choice(ps2_trivia)
        question = trivia_q["question"]
        correct_answer = trivia_q["correct"]
        all_options = trivia_q["options"]
        wrong_options = [opt for opt in all_options if opt != correct_answer]
    
    # Certificar-se de que as opções erradas não contêm a resposta certa
    wrong_options = [opt for opt in wrong_options if str(opt) != str(correct_answer)]
    
    # Limitar a 3 opções erradas
    if len(wrong_options) > 3:
        wrong_options = wrong_options[:3]
    
    # Adicionar respostas para termos 4 opções
    while len(wrong_options) < 3:
        if question_type == "year":
            new_wrong = str(random.randint(1997, 2010))
            if new_wrong != str(correct_answer) and new_wrong not in wrong_options:
                wrong_options.append(new_wrong)
        else:
            break  # Evitar loop infinito se não conseguir gerar opções diferentes
    
    # Misturar as opções
    options = wrong_options + [correct_answer]
    random.shuffle(options)
    
    return {
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "game_title": random_game['title'],
        "question_type": question_type
    }

# Função para selecionar uma opção no quiz
def select_option(option):
    st.session_state["selected_option"] = option
    st.session_state["answered"] = True
    
# Função para avançar para próxima pergunta
def next_question():
    st.session_state["quiz_question_count"] += 1
    st.session_state["current_question"] = {}  # Limpar pergunta atual para gerar nova
    st.session_state["answered"] = False
    st.session_state["selected_option"] = None

# Função para reiniciar o quiz
def restart_quiz():
    st.session_state["quiz_score"] = 0
    st.session_state["quiz_question_count"] = 0
    st.session_state["current_question"] = {}
    st.session_state["answered"] = False
    st.session_state["selected_option"] = None

# Função para voltar ao modo chat
def back_to_chat():
    st.session_state["quiz_mode"] = False
    st.session_state["quiz_started"] = False
    restart_quiz()  # Limpa o estado do quiz

# Função para mostrar o quiz
def show_quiz(df):
    st.header("🎮 Quiz de PlayStation 2")
    
    # Inicialização
    if st.session_state["quiz_question_count"] >= st.session_state["max_questions"]:
        # Quiz terminado, mostrar resultado final
        st.success(f"Quiz concluído! Sua pontuação: {st.session_state['quiz_score']}/{st.session_state['max_questions']}")
        
        # Mensagens personalizadas com base na pontuação
        score_percentage = (st.session_state['quiz_score'] / st.session_state['max_questions']) * 100
        
        if score_percentage == 100:
            st.balloons()
            st.write("🏆 Perfeito! Você é um verdadeiro mestre em PlayStation 2! Impressionante!")
        elif score_percentage >= 80:
            st.write("🥇 Excelente! Você realmente conhece muito bem os jogos de PS2!")
        elif score_percentage >= 60:
            st.write("🥈 Bom trabalho! Você tem um conhecimento sólido sobre PS2!")
        elif score_percentage >= 40:
            st.write("🥉 Não foi mal! Continue explorando o universo dos jogos de PS2!")
        else:
            st.write("😊 Há muito mais para descobrir sobre o PS2! Continue aprendendo!")
        
        # Botão para reiniciar ou voltar
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Jogar Novamente"):
                restart_quiz()
        with col2:
            if st.button("Voltar ao Chat"):
                back_to_chat()
                
    else:
        # Mostrar progresso
        st.progress((st.session_state["quiz_question_count"]) / st.session_state["max_questions"])
        st.write(f"Pergunta {st.session_state['quiz_question_count'] + 1} de {st.session_state['max_questions']}")
        st.write(f"Pontuação atual: {st.session_state['quiz_score']}")
        
        # Gerar nova pergunta se necessário
        if not st.session_state["current_question"]:
            st.session_state["current_question"] = generate_quiz_question(df)
        
        question = st.session_state["current_question"]
        
        # Exibir a pergunta
        st.subheader(question["question"])
        
        # Se já respondeu, desativar botões e mostrar feedback
        if st.session_state["answered"]:
            selected = st.session_state["selected_option"]
            
            # Verificar resposta
            if selected == question["correct_answer"]:
                st.success("✅ Correto!")
                if "answer_processed" not in st.session_state:
                    st.session_state["quiz_score"] += 1
                    st.session_state["answer_processed"] = True
            else:
                st.error(f"❌ Incorreto! A resposta certa é: {question['correct_answer']}")
            
            # Adicionar informação extra sobre o jogo após a resposta
            if question["question_type"] != "trivia":
                st.info(f"Sobre '{question['game_title']}': {df[df['title'] == question['game_title']]['description'].values[0]}")
            
            # Botão para próxima pergunta
            if st.button("Próxima Pergunta"):
                st.session_state.pop("answer_processed", None)  # Remover flag de resposta processada
                next_question()
                
        else:
            # Exibir as opções como botões grandes
            cols = st.columns(2)
            
            for i, option in enumerate(question["options"]):
                col_idx = i % 2
                with cols[col_idx]:
                    st.button(
                        f"{chr(65+i)}. {option}", 
                        key=f"opt_{i}", 
                        on_click=select_option,
                        args=(option,),
                        use_container_width=True
                    )

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
            model="gemini-2.0-flash-001",
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
    
    # Botão de Quiz na sidebar
    st.header("📝 Modo Quiz")
    if st.button("Iniciar Quiz de PS2"):
        st.session_state["quiz_mode"] = True
        st.session_state["quiz_started"] = True

# Verificar se estamos no modo quiz
if "quiz_mode" not in st.session_state:
    st.session_state["quiz_mode"] = False

# Inicializar o sistema
qa_chain = initialize_qa_system()

# Mostrar Quiz ou Chat com base no modo atual
# Mostrar Quiz ou Chat com base no modo atual
if st.session_state.get("quiz_mode", False):
    initialize_quiz()
    if "games_df" in st.session_state:
        show_quiz(st.session_state["games_df"])
    else:
        st.error("Não foi possível carregar os dados necessários para o Quiz.")
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