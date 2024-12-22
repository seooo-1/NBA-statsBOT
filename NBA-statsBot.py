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

# 데이터와 모델 로드
data = fetch_nba_data()
model = load_model()

if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=[])

for content in st.session_state.chat_session.history:
    with st.chat_message("ai" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)

if prompt := st.chat_input("메시지를 입력하세요."):    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("ai"):
        response = st.session_state.chat_session.send_message(prompt)        
        st.markdown(response.text)