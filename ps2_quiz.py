import random
import pandas as pd
import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

class PS2Quiz:
    """
    A class to handle the PlayStation 2 quiz functionality with a record system
    """
    
    def __init__(self, games_df: pd.DataFrame, max_questions: int = 5):
        """
        Initialize the quiz module
        
        Args:
            games_df: DataFrame containing PS2 games information
            max_questions: Maximum number of questions in a quiz session
        """
        self.games_df = games_df
        self.max_questions = max_questions
        self.records_file = "ps2_quiz_records.json"
        self.initialize_session_state()
        self.load_records()
    
    def initialize_session_state(self) -> None:
        """Initialize or reset the session state variables for the quiz"""
        if "quiz_score" not in st.session_state:
            st.session_state["quiz_score"] = 0
        if "quiz_question_count" not in st.session_state:
            st.session_state["quiz_question_count"] = 0
        if "max_questions" not in st.session_state:
            st.session_state["max_questions"] = self.max_questions
        if "current_question" not in st.session_state:
            st.session_state["current_question"] = {}
        if "answered" not in st.session_state:
            st.session_state["answered"] = False
        if "selected_option" not in st.session_state:
            st.session_state["selected_option"] = None
        if "player_name" not in st.session_state:
            st.session_state["player_name"] = ""
        if "show_record_form" not in st.session_state:
            st.session_state["show_record_form"] = False
        if "records" not in st.session_state:
            st.session_state["records"] = []
        if "record_saved" not in st.session_state:
            st.session_state["record_saved"] = False
    
    def load_records(self) -> None:
        """Load saved records from file"""
        try:
            if os.path.exists(self.records_file):
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    st.session_state["records"] = json.load(f)
            else:
                st.session_state["records"] = []
        except Exception as e:
            st.error(f"Erro ao carregar recordes: {str(e)}")
            st.session_state["records"] = []
    
    def save_record(self, player_name: str) -> None:
        """Save a new record to the records file, accumulating points for same player"""
        if not player_name:
            return
            
        # Dados do quiz atual
        current_score = st.session_state["quiz_score"]
        current_max = st.session_state["max_questions"]
        
        # Verificar se já existe um registro para este jogador
        player_exists = False
        for i, record in enumerate(st.session_state.get("records", [])):
            if record.get("player_name") == player_name:
                player_exists = True
                # Atualizar o registro existente somando a pontuação
                old_score = record.get("score", 0)
                old_max = record.get("max_score", 0)
                
                # Somar pontuações
                new_score = old_score + current_score
                new_max = old_max + current_max
                new_percentage = (new_score / new_max) * 100 if new_max > 0 else 0
                
                # Atualizar o registro
                st.session_state["records"][i] = {
                    "player_name": player_name,
                    "score": new_score,
                    "max_score": new_max,
                    "percentage": new_percentage,
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "session_id": str(id(self))
                }
                st.success(f"Pontuação de {player_name} atualizada! Total: {new_score}/{new_max} ({new_percentage:.1f}%)")
                break
                
        # Se não existe, criar um novo registro
        if not player_exists:
            try:
                # Create new record
                new_record = {
                    "player_name": player_name,
                    "score": current_score,
                    "max_score": current_max,
                    "percentage": (current_score / current_max) * 100 if current_max > 0 else 0,
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "session_id": str(id(self))
                }
                
                # Add to records list
                st.session_state["records"].append(new_record)
                st.success(f"Novo jogador {player_name} registrado com pontuação {current_score}/{current_max}!")
            except Exception as e:
                st.error(f"Erro ao registrar novo jogador: {str(e)}")
            
        # Ordenar registros por pontuação (percentual)
        st.session_state["records"] = sorted(
            st.session_state["records"], 
            key=lambda x: (x["percentage"], x["score"]), 
            reverse=True
        )
        
        # Limitar a 10 registros
        if len(st.session_state["records"]) > 10:
            st.session_state["records"] = st.session_state["records"][:10]
        
        try:
            # Salvar no arquivo
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state["records"], f, ensure_ascii=False, indent=2)
                
            st.session_state["show_record_form"] = False
            st.session_state["record_saved"] = True
            st.session_state["last_saved_score"] = {
                "player": player_name,
                "score": current_score,
                "max": current_max,
                "session_id": str(id(self))
            }
            
        except Exception as e:
            st.error(f"Erro ao salvar recordes: {str(e)}")
    
    def generate_question(self) -> Dict[str, Any]:
        """
        Generate a random quiz question about PS2 games
        
        Returns:
            Dictionary containing question details
        """
        # Verificar se temos jogos suficientes
        if len(self.games_df) < 5:
            # Fallback para perguntas de trivia se não houver jogos suficientes
            return self._generate_trivia_question()
        
        # Select a random game
        random_game = self.games_df.sample(1).iloc[0]
        
        # Decide question type with weighted probabilities
        question_types = ["developer", "year", "genre", "character", "trivia"]
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        question_type = random.choices(question_types, weights=weights, k=1)[0]
        
        if question_type == "developer":
            return self._generate_developer_question(random_game)
        elif question_type == "year":
            return self._generate_year_question(random_game)
        elif question_type == "genre":
            return self._generate_genre_question(random_game)
        elif question_type == "character":
            # If character question generation fails, try another type
            character_question = self._generate_character_question(random_game)
            if character_question:
                return character_question
            # Fallback to another question type
            return self.generate_question()
        elif question_type == "trivia":
            return self._generate_trivia_question()
    
    def _generate_developer_question(self, game: pd.Series) -> Dict[str, Any]:
        """Generate a question about game developer"""
        question = f"Quem desenvolveu o jogo '{game['title']}'?"
        correct_answer = str(game['developer'])
        
        # Generate wrong options (unique developers)
        wrong_options = []
        dev_samples = self.games_df['developer'].sample(min(4, len(self.games_df)))
        for dev in dev_samples:
            if str(dev) != correct_answer and dev not in wrong_options:
                wrong_options.append(str(dev))
        
        # Ensure we have at least some wrong options
        while len(wrong_options) < 3:
            wrong_options.append(f"Desenvolvedor Fictício {len(wrong_options)}")
            
        wrong_options = wrong_options[:3]  # Limit to 3
        
        return self._format_question(question, correct_answer, wrong_options, game)
    
    def _generate_year_question(self, game: pd.Series) -> Dict[str, Any]:
        """Generate a question about release year"""
        question = f"Em que ano foi lançado '{game['title']}'?"
        
        # Ensure year is a string
        if pd.isna(game['year']):
            # If year is missing, generate a trivia question instead
            return self._generate_trivia_question()
            
        correct_answer = str(int(game['year']))
        
        # Generate years close to the correct one
        year_int = int(game['year'])
        year_range = list(range(year_int - 3, year_int + 4))
        wrong_options = [str(year) for year in year_range if str(year) != correct_answer]
        wrong_options = random.sample(wrong_options, min(3, len(wrong_options)))
        
        # Ensure we have at least 3 wrong options
        while len(wrong_options) < 3:
            new_year = random.randint(1995, 2010)
            if str(new_year) != correct_answer and str(new_year) not in wrong_options:
                wrong_options.append(str(new_year))
        
        return self._format_question(question, correct_answer, wrong_options, game)
    
    def _generate_genre_question(self, game: pd.Series) -> Dict[str, Any]:
        """Generate a question about game genre"""
        question = f"Qual é o gênero de '{game['title']}'?"
        
        # Check if genre exists
        if pd.isna(game['genre']):
            # If genre is missing, generate a trivia question instead
            return self._generate_trivia_question()
            
        correct_answer = str(game['genre'])
        
        # Get unique genres for wrong options
        all_genres = list(self.games_df['genre'].dropna().unique())
        wrong_options = [str(genre) for genre in all_genres if str(genre) != correct_answer]
        
        if len(wrong_options) >= 3:
            wrong_options = random.sample(wrong_options, 3)
        else:
            # Fallback genres if there aren't enough unique ones
            fallback_genres = ["Ação", "Aventura", "RPG", "Estratégia", "Corrida", "Esportes", "Puzzle", "Simulação"]
            for genre in fallback_genres:
                if genre != correct_answer and genre not in wrong_options:
                    wrong_options.append(genre)
                if len(wrong_options) >= 3:
                    break
            
            wrong_options = wrong_options[:3]  # Limit to 3
        
        return self._format_question(question, correct_answer, wrong_options, game)
    
    def _generate_character_question(self, game: pd.Series) -> Optional[Dict[str, Any]]:
        """Generate a question about game characters"""
        # Check if character data exists and is usable
        if 'characters' not in game or pd.isna(game.get('characters')) or len(str(game['characters'])) <= 3:
            return None
            
        question = f"Qual destes personagens NÃO aparece em '{game['title']}'?"
        
        # Get characters from the current game
        characters = str(game['characters']).split(", ")
        
        # Find characters from other games
        other_characters = []
        for _, other_game in self.games_df.sample(min(5, len(self.games_df))).iterrows():
            if other_game['title'] != game['title'] and 'characters' in other_game and pd.notna(other_game.get('characters')):
                other_chars = str(other_game['characters']).split(", ")
                if other_chars:
                    other_characters.extend(other_chars)
        
        if other_characters:
            # Character that is NOT in the game (correct answer)
            correct_answer = random.choice(other_characters)
            
            # Characters from the game (wrong options)
            if len(characters) >= 3:
                wrong_options = random.sample(characters, 3)
            else:
                # If not enough characters in the game, add other wrong ones
                wrong_options = characters.copy()
                
                # Add fake characters if we don't have enough
                fake_characters = ["Personagem Fictício", "Personagem Inexistente", "Herói Imaginário"]
                for i in range(3 - len(wrong_options)):
                    wrong_options.append(fake_characters[i % len(fake_characters)])
            
            return self._format_question(question, correct_answer, wrong_options, game)
        
        # Return None if we couldn't generate a character question
        return None
    
    def _generate_trivia_question(self) -> Dict[str, Any]:
        """Generate a general PS2 trivia question"""
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
            },
            {
                "question": "Qual destes acessórios NÃO foi lançado oficialmente para o PlayStation 2?",
                "correct": "PlayStation VR",
                "options": ["PlayStation VR", "EyeToy", "Guitar Hero Controller", "DDR Dance Pad"]
            },
            {
                "question": "Qual empresa fabricou o PlayStation 2?",
                "correct": "Sony",
                "options": ["Sony", "Microsoft", "Nintendo", "Sega"]
            },
            {
                "question": "Qual foi o último jogo oficialmente lançado para PlayStation 2?",
                "correct": "Pro Evolution Soccer 2014",
                "options": ["Pro Evolution Soccer 2014", "FIFA 14", "Final Fantasy XII", "God of War II"]
            }
        ]
        
        # Select a random trivia question
        trivia_q = random.choice(ps2_trivia)
        question = trivia_q["question"]
        correct_answer = trivia_q["correct"]
        all_options = trivia_q["options"]
        
        # Separate wrong options
        wrong_options = [opt for opt in all_options if opt != correct_answer]
        
        return {
            "question": question,
            "options": all_options.copy(),  # Already shuffled in the trivia data
            "correct_answer": correct_answer,
            "game_title": "PlayStation 2 Trivia",
            "question_type": "trivia"
        }
    
    def _format_question(self, question: str, correct_answer: str, 
                         wrong_options: List[str], game: pd.Series) -> Dict[str, Any]:
        """Format the question data into a standardized dictionary"""
        # Ensure we have exactly 3 wrong options
        wrong_options = wrong_options[:3]
        
        # Create all options including the correct answer
        options = wrong_options + [correct_answer]
        random.shuffle(options)  # Randomize the order
        
        return {
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "game_title": game['title'],
            "question_type": game.get('question_type', 'game')
        }
    
    def select_option(self, option: str) -> None:
        """Handle option selection"""
        st.session_state["selected_option"] = option
        st.session_state["answered"] = True
    
    def next_question(self) -> None:
        """Advance to the next question"""
        st.session_state["quiz_question_count"] += 1
        st.session_state["current_question"] = {}  # Clear current question
        st.session_state["answered"] = False
        st.session_state["selected_option"] = None
        st.session_state.pop("answer_processed", None)  # Remove processed flag
    
    def restart_quiz(self) -> None:
        """Reset the quiz to start over"""
        st.session_state["quiz_score"] = 0
        st.session_state["quiz_question_count"] = 0
        st.session_state["current_question"] = {}
        st.session_state["answered"] = False
        st.session_state["selected_option"] = None
    
    def render_quiz(self) -> None:
        """Display the quiz interface"""
        st.header("🎮 Quiz de PlayStation 2")
        
        # Check if quiz is completed
        if st.session_state["quiz_question_count"] >= st.session_state["max_questions"]:
            self._render_quiz_results()
        else:
            self._render_quiz_question()

    def _render_quiz_results(self) -> None:
        """Display the final quiz results and record system"""
        st.success(f"Quiz concluído! Sua pontuação: {st.session_state['quiz_score']}/{st.session_state['max_questions']}")
        
        # Personalized messages based on score
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
        
        # Verificar se já temos um nome de jogador salvo na sessão
        player_name = st.session_state.get("player_name", "")
        
        # Se já temos um nome definido, salvar automaticamente
        if player_name:
            try:
                self.save_record(player_name)
                st.success(f"Pontuação de {player_name} registrada com sucesso!")
            except Exception as e:
                st.error(f"Erro ao salvar pontuação: {str(e)}")
        else:
            # Se não temos nome definido, mostrar o formulário para entrar com o nome
            if st.session_state.get("show_record_form", False):
                with st.form("record_form"):
                    st.subheader("Salvar seu recorde")
                    player_name = st.text_input("Seu nome:", value="")
                    
                    submitted = st.form_submit_button("Salvar")
                    if submitted and player_name:
                        self.save_record(player_name)
            else:
                # Show option to save record
                if st.button("Registrar meu recorde"):
                    st.session_state["show_record_form"] = True
                    st.experimental_rerun()
                
        # Show leaderboard if we have records
        if st.session_state.get("records") and len(st.session_state["records"]) > 0:
            st.subheader("🏆 Recordes")
            records_df = pd.DataFrame(st.session_state["records"])
            
            # Format the table for display
            records_df = records_df[["player_name", "score", "max_score", "percentage", "date"]]
            records_df.columns = ["Jogador", "Pontuação", "Total", "Percentual", "Data"]
            records_df["Percentual"] = records_df["Percentual"].apply(lambda x: f"{x:.1f}%")
            
            st.table(records_df.head(5))  # Show only top 5 for cleaner UI
            
            if len(st.session_state["records"]) > 5:
                with st.expander("Ver todos os recordes"):
                    st.table(records_df)
        
        # Buttons to restart or return to chat
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Jogar Novamente"):
                st.session_state["show_record_form"] = False
                self.restart_quiz()
        with col2:
            if st.button("Voltar ao Chat"):
                st.session_state["quiz_mode"] = False
                st.session_state["quiz_started"] = False
                st.session_state["show_record_form"] = False
                self.restart_quiz()

    def _render_quiz_question(self) -> None:
        """Display the current quiz question"""
        # Show progress
        st.progress((st.session_state["quiz_question_count"]) / st.session_state["max_questions"])
        st.write(f"Pergunta {st.session_state['quiz_question_count'] + 1} de {st.session_state['max_questions']}")
        st.write(f"Pontuação atual: {st.session_state['quiz_score']}")
        
        # Generate new question if needed
        if not st.session_state["current_question"]:
            try:
                st.session_state["current_question"] = self.generate_question()
            except Exception as e:
                st.error(f"Erro ao gerar pergunta: {str(e)}")
                st.session_state["current_question"] = self._generate_trivia_question()  # Fallback
        
        question = st.session_state["current_question"]
        
        # Display question
        st.subheader(question["question"])
        
        # If already answered, show feedback
        if st.session_state["answered"]:
            selected = st.session_state["selected_option"]
            
            # Check answer
            if selected == question["correct_answer"]:
                st.success("✅ Correto!")
                if "answer_processed" not in st.session_state:
                    st.session_state["quiz_score"] += 1
                    st.session_state["answer_processed"] = True
            else:
                st.error(f"❌ Incorreto! A resposta certa é: {question['correct_answer']}")
            
            # Show extra information about the game
            if question["question_type"] != "trivia":
                game_info = self.games_df[self.games_df['title'] == question['game_title']]['description']
                if len(game_info) > 0 and not pd.isna(game_info.iloc[0]):
                    st.info(f"Sobre '{question['game_title']}': {game_info.iloc[0]}")
            
            # Button for next question
            if st.button("Próxima Pergunta"):
                self.next_question()
        else:
            # Display options as buttons
            cols = st.columns(2)
            
            for i, option in enumerate(question["options"]):
                col_idx = i % 2
                with cols[col_idx]:
                    if option:  # Ensure option is not None or empty
                        st.button(
                            f"{chr(65+i)}. {option}", 
                            key=f"opt_{i}", 
                            on_click=self.select_option,
                            args=(option,),
                            use_container_width=True
                        )


# Function to use in the main application
def create_ps2_quiz(games_df: pd.DataFrame, max_questions: int = 5) -> PS2Quiz:
    """
    Create a PS2 quiz instance
    
    Args:
        games_df: DataFrame containing PS2 games information
        max_questions: Maximum number of questions per quiz
        
    Returns:
        PS2Quiz instance
    """
    return PS2Quiz(games_df, max_questions)