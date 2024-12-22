import requests
import json
import pandas as pd
import re
import google.generativeai as genai
import streamlit as st

# 설정: Gemini API 키 입력
YOUR_API_KEY = 'AIzaSyDf58dSIRrAxfgvAW5LIALU3DqkTaV-c-U'
genai.configure(api_key=YOUR_API_KEY)

# 제목 표시
st.title("NBA-StatsBot")

# NBA 데이터 가져오기 함수
@st.cache_data
def fetch_nba_data():
    url = 'https://stats.nba.com/stats/leagueLeaders?LeagueID=00&PerMode=PerGame&Scope=S&Season=2024-25&SeasonType=Regular%20Season&StatCategory=PTS'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    json_info = json.loads(response.text)
    stats = json_info["resultSet"]["rowSet"]
    
    # 데이터프레임 생성
    df = pd.DataFrame(columns=[
        "RANK", "PLAYER", "TEAM", "GP", "MIN", "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", 
        "DREB", "REB", "AST", "STL", "BLK", "TOV", "PTS", "EFF"
    ])
    for stat in stats:
        df.loc[len(df)] = {
            "RANK": stat[1],
            "PLAYER": stat[2],
            "TEAM": stat[4],
            "GP": stat[5],
            "MIN": stat[6],
            "FGM": stat[7],
            "FGA": stat[8],
            "FG_PCT": stat[9],
            "FG3M": stat[10],
            "FG3A": stat[11],
            "FG3_PCT": stat[12],
            "FTM": stat[13],
            "FTA": stat[14],
            "FT_PCT": stat[15],
            "OREB": stat[16],
            "DREB": stat[17],
            "REB": stat[18],
            "AST": stat[19],
            "STL": stat[20],
            "BLK": stat[21],
            "TOV": stat[22],
            "PTS": stat[23],
            "EFF": stat[24],
        }
    df.index += 1
    return df

# 모델 로드 함수
@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Model loaded...")
    return model

# 자연어 질문에서 선수 이름과 스탯 추출
def parse_question(question, data):
    # 선수 이름 추출
    player_names = data['PLAYER'].tolist()
    player_name = next((name for name in player_names if name.lower() in question.lower()), None)

    # 스탯 이름 추출
    stat_keywords = {
        "PTS": ["points", "득점", "점수"],
        "REB": ["rebounds", "리바운드"],
        "AST": ["assists", "어시스트"],
        "STL": ["steals", "스틸"],
        "BLK": ["blocks", "블록"],
        "FG_PCT": ["field goal percentage", "필드골 성공률"],
        "FG3M": ["three-pointers made", "3점슛"],
        "FT_PCT": ["free throw percentage", "자유투 성공률"]
    }
    stat_name = next((key for key, keywords in stat_keywords.items() if any(keyword in question.lower() for keyword in keywords)), None)

    return player_name, stat_name

# 데이터와 모델 로드
data = fetch_nba_data()
model = load_model()

# 세션 상태 초기화
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = model.start_chat(history=[])

# 이전 채팅 기록 표시
for content in st.session_state.chat_session.history:
    with st.chat_message("ai" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)

# 사용자 입력 처리
if prompt := st.chat_input("선수와 알고 싶은 내용을 물어보세요! (예: LeBron James의 점수를 알려줘)"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # 질문 파싱하여 선수와 스탯 추출
    player_name, stat_name = parse_question(prompt, data)

    # 결과 처리
    if player_name and stat_name:
        if player_name in data['PLAYER'].values:
            player_stats = data[data['PLAYER'] == player_name]
            if stat_name in player_stats.columns:
                stat_value = player_stats[stat_name].values[0]
                user_query = f"{player_name}의 {stat_name}은 {stat_value}입니다. 이는 24-25 시즌 데이터이며 신뢰할 수 있는 데이터이니 이를 가정하고 해당 데이터에 대한 분석을 해주세요"
            else:
                user_query = f"{stat_name}은(는) 데이터에 없는 스탯입니다."
        else:
            user_query = f"{player_name}은(는) 데이터에 없는 선수입니다."
    else:
        user_query = "질문에서 선수 이름이나 스탯을 찾을 수 없습니다. 다시 질문해 주세요!"

    # Gemini 모델 응답 생성
    with st.chat_message("ai"):
        response = st.session_state.chat_session.send_message(user_query)
        st.markdown(response.text)
