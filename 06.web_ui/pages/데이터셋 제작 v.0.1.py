# scipy 모듈 설치를 위해 아래 명령어 중 하나를 streamlit 가상환경 터미널에서 실행
# pip install -r 06.web_ui/requirements.txt
# pip install -r requirements.txt

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import json
import pymysql
import holidays
import time
from sqlalchemy import create_engine, text
from datetime import datetime
import os

st.set_page_config(layout="wide")
#st.image("./images/accident-5167244_2.jpg", use_container_width=True)
st.image("./06.web_ui/images/accident-5167244_2.jpg", use_container_width=True)
st.title("데이터셋 생성 마법사 🧙")
st.write("")

def database_connector():
    # 데이터베이스 연결
    with open("./db_config.json", "r") as f:
        config = json.load(f)

    DB_USER = config["DB_USER"]
    DB_PASSWORD = config["DB_PASSWORD"]
    DB_HOST = config["DB_HOST"]
    DB_PORT = config["DB_PORT"]
    DB_NAME = config["DB_NAME"]

    engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # 1️⃣ "데이터 연결중..." 메시지를 표시
    loading_message = st.empty()
    loading_message.info("**데이터 연결중...**")

    try:
        time.sleep(2)  # 시뮬레이션 (연결 시간)
        loading_message.empty()
        engine = create_engine(engine_url)
        st.success("**MySQL 데이터베이스에 성공적으로 연결되었습니다.**")
    except Exception as e:
        st.error(f"**데이터베이스 연결 오류: {e}**")
        st.stop()
    return engine

def database_disconnector(engine):
    # 데이터베이스 연결 종료
    if 'engine' in locals() and engine:
        engine.dispose()
    st.write("\n**데이터베이스 연결이 종료되었습니다.**")

# --- 데이터베이스에서 데이터 로드하는 함수 ---
def load_table_to_df(table_name, engine):
    """**지정된 테이블에서 모든 데이터를 Pandas DataFrame으로 로드합니다.**"""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(text(query), engine.connect())
        st.write(f"**'{table_name}' 테이블 로드 완료. {len(df)} 행.**")
        # 날짜/시간 컬럼 타입 변환 (필요시)
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
        if 'accident_date' in df.columns:
            df['accident_date'] = pd.to_datetime(df['accident_date'])
        if 'weather_date' in df.columns:
            df['weather_date'] = pd.to_datetime(df['weather_date'])
        # start_time, end_time, weather_time은 문자열로 로드될 수 있으므로, 필요시 time 객체로 변환
        return df
    except Exception as e:
        st.info(f"**'{table_name}' 테이블 로드 중 오류 발생: {e}**")
        return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환

def get_base_df():
    base_df = pd.DataFrame(columns=[
        'date', 'region', 'accident_count', 'injury_count', 'death_count', 'game_count',
        'sports_type', 'temperature', 'precipitation', 'snow_depth', 'weather_condition',
        'is_post_season', 'is_hometeam_win', 'is_holiday', 'weekday', 'audience',
        'game_start_time', 'game_end_time'
    ])
    return base_df.copy()

def get_dataset(TARGET_REGION,TARGET_SPORTS,TARGET_WEEKDAYS,TARGET_DATES):
    # --- 데이터 로드 ---
    engine = database_connector()
    stadium_df = load_table_to_df('stadium', engine)
    sports_game_df = load_table_to_df('sports_game', engine)
    traffic_accident_df = load_table_to_df('traffic_accident', engine)
    weather_df = load_table_to_df('weather', engine)
    date_range = pd.date_range(start=TARGET_DATES[0], end=TARGET_DATES[1], freq='D')
    final_df = get_base_df()
    for region in TARGET_REGION:
        temp_df = get_base_df()
        temp_df = pd.DataFrame({'date': date_range})
        temp_df['region'] = region
        
        # 스타디움 정보 가져오기
        if not stadium_df.empty:
            stadiums_in_target_region = stadium_df[stadium_df['region'] == region]
            stadium_codes_in_region = stadiums_in_target_region['stadium_code'].unique().tolist()
            #print(f"\n{TARGET_REGION} 내 경기장 코드: {stadium_codes_in_region}")
        else:
            stadium_codes_in_region = []
            #print(f"\n{TARGET_REGION} 내 경기장 정보 없음 또는 stadium 테이블 로드 실패.")
            
        # 스포츠경기 정보 가져오기
        if not sports_game_df.empty and stadium_codes_in_region:
            games_in_region_df = sports_game_df[sports_game_df['stadium_code'].isin(stadium_codes_in_region)]
            games_in_region_df = games_in_region_df.rename(columns={'game_date': 'date'})
            #print(games_in_region_df)
            if not games_in_region_df.empty:
                games_in_region_df["match_type"] = (
                    games_in_region_df["match_type"]
                        .replace({"페넌트레이스": "정규시즌",
                                "순위결정전": "정규시즌",
                                "순위결정정": "정규시즌",   # 오타까지 함께 처리
                                '조별리그' : "정규시즌",
                                "0": "정규시즌"})
                        # ➋ 라운드 표기(1R ~ 33R 등) → 정규시즌
                        .str.replace(r"^\d+R$", "정규시즌", regex=True)
                )
                games_in_region_df["match_type"] = (
                    games_in_region_df["match_type"]
                        .replace({'와일드카드':"포스트시즌",
                                '준플레이오프':"포스트시즌", 
                                '플레이오프':"포스트시즌", 
                                '한국시리즈':"포스트시즌",
                                '파이널 라운드A':"포스트시즌",
                                '파이널 라운드B':"포스트시즌",
                                '챔피언결정전':"포스트시즌", 
                                '준결승':"포스트시즌", 
                                '결승':"포스트시즌"})
                )
                game_summary_df = games_in_region_df.groupby('date').agg(
                    game_count=('stadium_code', 'count'),
                    sports_types_list=('sports_type', lambda x: list(set(x))),
                    is_post_season_list=('match_type', lambda x: 1 if any('포스트시즌' in str(mt).lower() for mt in x) else 0),
                    game_start_time_agg=('start_time', 'min'),
                    game_end_time_agg=('end_time', 'max'),
                    is_hometeam_win_agg=('home_team_win', 'max'),
                    audience_agg=('audience', 'sum')
                ).reset_index()
                game_summary_df['sports_type'] = game_summary_df['sports_types_list'].apply(lambda x: ','.join(sorted(x)) if x else '없음')
                game_summary_df['is_post_season'] = game_summary_df['is_post_season_list'].astype(int)
                game_summary_df['game_start_time'] = game_summary_df['game_start_time_agg']
                game_summary_df['game_end_time'] = game_summary_df['game_end_time_agg']
                game_summary_df['is_hometeam_win'] = game_summary_df['is_hometeam_win_agg'].astype(int)
                game_summary_df['audience'] = game_summary_df['audience_agg'].astype(int)
                game_summary_df = game_summary_df[['date', 'game_count', 'sports_type', 'is_post_season', 'game_start_time', 'game_end_time', 'is_hometeam_win', 'audience']]
                temp_df = pd.merge(temp_df, game_summary_df, on='date', how='left')
            else:
                print(f"{region} 내 해당 기간 경기 정보 없음.")
        else:
            print("sports_game_df 로드 실패 또는 대상 지역 내 경기장 없음.")

        temp_df['game_count'] = temp_df['game_count'].fillna(0).astype(int)
        temp_df['sports_type'] = temp_df['sports_type'].fillna('없음')
        temp_df['is_post_season'] = temp_df['is_post_season'].fillna(0).astype(int)
        temp_df['game_start_time'] = temp_df['game_start_time'].fillna(pd.NA) # 또는 적절한 기본값 (예: '정보없음')
        temp_df['game_end_time'] = temp_df['game_end_time'].fillna(pd.NA)   # 또는 적절한 기본값
        temp_df['is_hometeam_win'] = temp_df['is_hometeam_win'].fillna(0).astype(int) # 경기가 없으면 홈팀 승리도 0
        temp_df['audience'] = temp_df['audience'].fillna(0).astype(int)
        
        # 교통사고 데이터 가져오기
        if not traffic_accident_df.empty:
            accidents_in_region_df = traffic_accident_df[traffic_accident_df['region'] == region]
            accidents_in_region_df = accidents_in_region_df.rename(columns={'accident_date': 'date'})
            if not accidents_in_region_df.empty:
                accident_summary_df = accidents_in_region_df.groupby('date').agg(
                    accident_count_sum=('accident_count', 'sum'),
                    injury_count_sum=('injury_count', 'sum'),
                    death_count_sum=('death_count', 'sum')
                ).reset_index()
                accident_summary_df = accident_summary_df.rename(columns={'accident_count_sum': 'accident_count', 'injury_count_sum': 'injury_count', 'death_count_sum': 'death_count'})
                
                temp_df = pd.merge(temp_df, accident_summary_df, on='date', how='left')
            else:
                print(f"{region} 내 해당 기간 교통사고 정보 없음.")
                temp_df['accident_count'] = 0
                temp_df['injury_count'] = 0
                temp_df['death_count'] = 0
        else:
            print("traffic_accident_df 로드 실패.")
            temp_df['accident_count'] = 0
            temp_df['injury_count'] = 0
            temp_df['death_count'] = 0
            
        temp_df['accident_count'] = temp_df['accident_count'].fillna(0).astype(int)
        temp_df['injury_count'] = temp_df['injury_count'].fillna(0).astype(int)
        temp_df['death_count'] = temp_df['death_count'].fillna(0).astype(int)
        
        # 날씨 데이터 가져오기
        if not weather_df.empty:
            mask = weather_df['region'].apply(lambda x: x in region)
            weather_region_df = weather_df[mask]
            weather_region_df = weather_region_df.rename(columns={'weather_date': 'date'})

            if not weather_region_df.empty:
                # 날씨 데이터는 하루에 여러 번 기록될 수 있으므로, 일별 집계 필요
                weather_summary_df = weather_region_df.groupby('date').agg(
                    temperature=('temperature', 'mean'),
                    precipitation=('precipitation', 'sum'),
                    snow_depth=('snow_depth', 'sum'),
                    avg_cloud_amount=('cloud_amount', 'mean') # 대표 날씨 상태 추론용
                ).reset_index()

                def get_weather_condition(row):
                    if (pd.isna(row['precipitation']) or pd.isna(row['snow_depth'])) and pd.isna(row['avg_cloud_amount']):
                        return '정보없음' # 데이터가 없는 경우
                    if row['precipitation'] > 0 or pd.isna(row['snow_depth']) > 0:
                        return '비'
                    elif pd.notna(row['avg_cloud_amount']):
                        if row['avg_cloud_amount'] >= 7: # (0-10 기준)
                            return '흐림'
                        elif row['avg_cloud_amount'] >= 3:
                            return '약간흐림' # 또는 '약간흐림' 등
                        else:
                            return '맑음'
                    return '정보없음' # 강수량 없고 구름 정보도 없는 경우
                    
                weather_summary_df['weather_condition'] = weather_summary_df.apply(get_weather_condition, axis=1)
                weather_summary_df = weather_summary_df[['date', 'temperature', 'precipitation', 'snow_depth', 'weather_condition']]
                temp_df = pd.merge(temp_df, weather_summary_df, on='date', how='left')
            else:
                print(f"{region} 내 해당 기간 날씨 정보 없음.")
                temp_df['temperature'] = np.nan
                temp_df['precipitation'] = np.nan
                temp_df['snow_depth'] = np.nan
                temp_df['weather_condition'] = '정보없음'
        else:
            print("weather_df 로드 실패.")
            temp_df['temperature'] = np.nan
            temp_df['precipitation'] = np.nan
            temp_df['snow_depth'] = np.nan
            temp_df['weather_condition'] = '정보없음'
        
        # 공휴일 및 주말 데이터 가져오기
        weekday_map_kr = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
        temp_df['weekday'] = temp_df['date'].dt.dayofweek.map(weekday_map_kr)
        try:
            import holidays
            # temp_df['date']에서 연도를 뽑아와서 unique 값으로 추출
            kr_holidays = holidays.country_holidays('KR', years=temp_df['date'].dt.year.unique().tolist())
            is_statutory_holiday = temp_df['date'].apply(lambda d: d in kr_holidays)
            is_saturday = (temp_df['weekday'] == '토')
            is_sunday = (temp_df['weekday'] == '일')
            # 3. 세 가지 조건을 OR 연산자로 결합하여 is_holiday 컬럼 생성
            # (하나라도 True이면 True -> 1, 모두 False이면 False -> 0)
            temp_df['is_holiday'] = (is_statutory_holiday | is_saturday | is_sunday).astype(int)
        except ImportError:
            print("`holidays` 라이브러리가 설치되지 않았습니다. `pip install holidays`로 설치해주세요. 'is_holiday'는 0으로 처리됩니다.")
            temp_df['is_holiday'] = 0
        except Exception as e:
            print(f"공휴일 정보 처리 중 오류: {e}. 'is_holiday'는 0으로 처리됩니다.")
            temp_df['is_holiday'] = 0

        final_df = pd.merge(final_df,temp_df,how='outer')
    final_df = final_df[final_df['weekday'].isin(TARGET_WEEKDAYS)]
    final_df = final_df[final_df['sports_type'].isin(TARGET_SPORTS)]
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    final_df['temperature'] = pd.to_numeric(final_df['temperature'], errors='coerce').round(1)
    final_df['precipitation'] = pd.to_numeric(final_df['precipitation'], errors='coerce').round(1)
    final_df['snow_depth'] = pd.to_numeric(final_df['precipitation'], errors='coerce').round(1)
    database_disconnector(engine)
    return final_df

# 세션 상태 초기화 함수
def initialize_session_state():
    defaults = {
        "tab0_current_step": 0,
        "tab1_current_step": 0,
        "region_search_term": "",
        "TARGET_REGION": [],
        "TARGET_SPORTS": [],
        "TARGET_WEEKDAYS": [],
        "TARGET_DATES": [],
        "final_df": None,
        "region_search_term_input": "",
        "df_filename": ""
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session():
    # 세션 상태를 초기화하고 앱을 다시 시작합니다.
    for key in st.session_state.keys():
        del st.session_state[key]

# 변수 초기화
START_DATE = pd.to_datetime("20230101") # 분석 시작일
END_DATE = pd.to_datetime("20241231")   # 분석 종료일
sports_list = ['야구','축구','배구','농구','없음']
weekday_list = ['월','화','수','목','금','토','일']
stadium_region_list = ['강원 강릉시', '강원 원주시', '강원 춘천시', '경기 고양시', '경기 김포시', '경기 수원시', '경기 안산시', '경기 안양시', 
                        '경기 의정부시', '경기 이천시', '경기 화성시', '경남 창원시', '경북 구미시', '경북 김천시', '경북 포항시', '광주 광산구', 
                        '광주 북구', '대구 동구', '대구 수성구', '대전 유성구', '대전 중구', '부산 동래구', '부산 연제구', '서울 구로구', 
                        '서울 마포구', '서울 송파구', '서울 양천구', '서울 중구', '울산 남구', '울산 중구', '인천 계양구', '인천 남동구', 
                        '인천 미추홀구', '인천 서구', '전북 군산시', '전북 전주시', '제주 제주시', '충남 천안시', '충북 청주시']
SELECT_ALL_KEYWORDS = ["전체", "전지역", "'전체'", "'전지역'"]


def display_step0_region_selection():
    st.subheader("1. 지역 선택")
    is_disabled = st.session_state.tab0_current_step > 0  # 0단계 이후에는 비활성화

    # 콜백 함수 정의
    def handle_search_input_change():
        search_term = st.session_state.region_search_term_input.strip().lower()
        lower_keywords = [kw.lower() for kw in SELECT_ALL_KEYWORDS]

        if search_term in lower_keywords:
            st.session_state.TARGET_REGION = list(stadium_region_list)
            st.session_state.region_search_term_input = "" # 검색창 비우기
            st.session_state.region_search_term = ""
        else:
            st.session_state.region_search_term = st.session_state.region_search_term_input

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.text_input(
            "**지역구 검색:** (지역구를 입력하거나 '전체' 또는 '전지역' 입력 시 모든 지역 선택)",
            key="region_search_term_input",
            on_change=handle_search_input_change,
            disabled=is_disabled
        )

        current_search = st.session_state.region_search_term.strip().lower()
        if current_search:
            filtered_regions = [r for r in stadium_region_list if current_search in r.lower()]
        else:
            filtered_regions = stadium_region_list

        options_for_multiselect = sorted(list(set(filtered_regions + st.session_state.TARGET_REGION)))

        st.multiselect(
            "**데이터셋을 구성할 지역명을 선택하세요.**",
            options=options_for_multiselect,
            key='TARGET_REGION',
            disabled=is_disabled,
            help="위 검색창에 지역명을 입력하여 목록을 줄이거나, '전체' 또는 '전지역'을 입력하여 모든 항목을 선택할 수 있습니다."
        )
    with col2:
        if st.session_state.tab0_current_step == 0:
            st.write("")
            st.write("")
            if st.button("다음 (스포츠 선택)", key="region_next"):
                if st.session_state.TARGET_REGION: # 지역이 선택되었는지 확인
                    st.session_state.tab0_current_step = 1
                    st.rerun()
                else:
                    st.warning("하나 이상의 지역을 선택해주세요.")

def display_step1_sports_selection():
    st.subheader("2. 스포츠 선택")
    is_disabled = st.session_state.tab0_current_step > 1 # 1단계 이후에는 비활성화

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.multiselect(
            "**데이터셋을 구성할 스포츠를 선택하세요.**",
            options=sports_list,
            key='TARGET_SPORTS',
            default=sports_list,
            disabled=is_disabled
        )
    with col2:
        if st.session_state.tab0_current_step == 1:
            st.write("")
            st.write("")
            if st.button("다음 (요일 선택)", key="sports_next"):
                if st.session_state.TARGET_SPORTS: # 스포츠가 선택되었는지 확인
                    st.session_state.tab0_current_step = 2
                    st.rerun()
                else:
                    st.warning("하나 이상의 스포츠를 선택해주세요.")

def display_step2_weekday_selection():
    st.subheader("3. 요일 선택")
    is_disabled = st.session_state.tab0_current_step > 2 # 2단계 이후에는 비활성화
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.multiselect(
            "**데이터셋을 구성할 요일들을 선택하세요**",
            options=weekday_list,
            key='TARGET_WEEKDAYS',
            default=weekday_list,
            disabled=is_disabled
        )
    with col2:
        if st.session_state.tab0_current_step == 2:
            st.write("")
            st.write("")
            if st.button("다음 (날짜 선택)", key="weekdays_next"):
                if st.session_state.TARGET_WEEKDAYS: # 요일이 선택되었는지 확인
                    st.session_state.tab0_current_step = 3
                    st.rerun()
                else:
                    st.warning("요일을 하나 이상 선택해주세요.")

def display_step3_dates_selection():
    st.subheader("4. 날짜 범위 선택")
    is_disabled = st.session_state.tab0_current_step > 3 # 3단계 이후에는 비활성화
    
    col1, col2 = st.columns([0.8,0.2])
    with col1:
        # st.date_input은 튜플 (시작일, 종료일) 또는 단일 날짜를 반환.
        # value는 2-element tuple로 시작일과 종료일을 지정
        st.date_input(
            "**분석할 날짜 범위를 선택하세요:**",
            value=(START_DATE, END_DATE), # 기본값으로 START_DATE, END_DATE 사용
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2024, 12, 31),
            key="TARGET_DATES",
            disabled=is_disabled
        )
    with col2:
        if st.session_state.tab0_current_step == 3:
            st.write("")
            st.write("")
            if st.button("다음 (데이터셋 생성)", key="dates_next"):
                if st.session_state.TARGET_DATES and len(st.session_state.TARGET_DATES) == 2:
                    # Pandas Timestamp 객체로 변환
                    start_dt = pd.to_datetime(st.session_state.TARGET_DATES[0])
                    end_dt = pd.to_datetime(st.session_state.TARGET_DATES[1])
                    if start_dt <= end_dt:
                        st.session_state.tab0_current_step = 4 # 모든 입력 완료
                        st.rerun()
                    else:
                        st.warning("시작일은 종료일보다 이전이거나 같아야 합니다.")
                else:
                    st.warning("정확한 날짜 범위를 선택해주세요.")

def display_step4_build_dataset():
    st.subheader("📊 최종 선택 결과 및 데이터셋 빌드")
    col1, col2 = st.columns([0.8,0.2])
    with col1:
        selected_region_display = ', '.join(st.session_state.get("TARGET_REGION", []))
        st.success(f"✔️ **선택 완료된 지역:** {selected_region_display}")
        selected_sports_display = ', '.join(st.session_state.get("TARGET_SPORTS", []))
        st.success(f"✔️ **선택 완료된 스포츠:** {selected_sports_display}")
        selected_weekday_display = ', '.join(st.session_state.get("TARGET_WEEKDAYS", []))
        st.success(f"✔️ **선택 완료된 요일:** {selected_weekday_display}")
        selected_dates_display = f"{st.session_state.TARGET_DATES[0].strftime('%Y-%m-%d')} ~ {st.session_state.TARGET_DATES[1].strftime('%Y-%m-%d')}"
        st.success(f"✔️ **선택 완료된 날짜 범위:** {selected_dates_display}")

        # 데이터셋 가져오기 (이전에 이미 가져왔다면 다시 가져오지 않음)
        if st.session_state.final_df is None:
            with st.spinner("데이터셋을 가져오는 중..."):
                st.session_state.final_df = get_dataset(
                    st.session_state.TARGET_REGION,
                    st.session_state.TARGET_SPORTS,
                    st.session_state.TARGET_WEEKDAYS,
                    st.session_state.TARGET_DATES
                )
        if st.session_state.final_df is not None:
            os.makedirs("./datas", exist_ok=True)
            st.session_state.df_filename = f"./datas/dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.final_df.to_csv(st.session_state.df_filename)
        else:
            st.error("데이터셋을 가져오는데 실패했습니다.")
    with col2:
        if st.session_state.tab0_current_step == 4:
            st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            if st.button("다음 (데이터셋 정보)", key="dataset_build_next"):
                if st.session_state.TARGET_WEEKDAYS: # 요일이 선택되었는지 확인
                    st.session_state.tab0_current_step = 5
                    st.rerun()
                else:
                    st.warning("요일을 하나 이상 선택해주세요.")
        pass

def display_reset_dataset():
    col1, col2 = st.columns([0.2,0.8])
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("처음부터 다시 선택하기", key="reset_all"):
            reset_session()
            st.rerun()
    with col2:
        pass

def display_step5_dataset_info():
    st.subheader("데이터셋 정보 및 결측치 확인")
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    if st.session_state.tab0_current_step >= 5:
        col1, col2 = st.columns([0.8,0.2])
        with col1:
            if st.session_state.final_df is not None:
                df = st.session_state.final_df.copy()
                st.write("\ndescribe DataFrame:")
                st.dataframe(df.describe())
                subcol1,subcol2 = st.columns([0.2,0.8])
                with subcol1:
                    st.write("\n컬럼별 결측치 갯수:")
                    st.dataframe(df.isnull().sum())
                with subcol2:
                    st.write("\n컬럼별 결측치 비율:")
                    na_pct = df.isna().mean().mul(100).sort_values(ascending=False)
                    if not na_pct.empty:
                        fig, ax = plt.subplots(figsize=(10, 5)) # figsize 조정 가능
                        sns.barplot(x=na_pct.index, y=na_pct.values, ax=ax, palette="viridis") # ax 전달, palette 추가
                        ax.tick_params(axis='x', rotation=45)
                        ax.set_ylabel("% Missing")
                        ax.set_xlabel("Columns") # X축 레이블 추가
                        ax.set_title("결측치 비율 (%)", fontsize=15) # 제목 폰트 크기
                        ax.grid(axis='y', linestyle='--') # y축 그리드만, 스타일 변경
                        plt.tight_layout() # 레이아웃 자동 조정
                        st.pyplot(fig) 
            else:
                st.warning("데이터셋이 없습니다. 먼저 데이터셋을 생성해주세요.")
        with col2:
            if st.session_state.tab0_current_step == 5:
                st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
                if st.button("다음 (히트맵)", key="dataset_info_next"):
                    st.session_state.tab0_current_step = 6
                    st.rerun()

def display_step6_dataset_heatmap():
    st.subheader("데이터셋 히트맵")
    if st.session_state.tab0_current_step >= 6:
        col1, col2 = st.columns([0.8,0.2])
        with col1:
            num_cols = ["accident_count","game_count","temperature","precipitation","audience","is_post_season","is_hometeam_win","is_holiday"]
            cat_cols  = ["region","sports_type","weekday"]
            df = st.session_state.final_df.copy()
           # 상관계수 히트맵
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
            ax.set_title("Spearman Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            if st.session_state.tab0_current_step == 6:
                st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
                if st.button("다음 (선형회귀선)", key="dataset_heatmap_next"):
                    st.session_state.tab0_current_step = 7
                    st.rerun()

def display_step7_dataset_linear():
    st.subheader("데이터셋 산점도 및 선형회귀선")
    if st.session_state.tab0_current_step >= 7:
        col1, col2 = st.columns([0.8,0.2])
        with col1:
            df = st.session_state.final_df.copy()
            num_cols = ["accident_count","game_count","temperature","precipitation","audience"]
            for col in num_cols:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.regplot(x=col, y="accident_count", data=df,
                            scatter_kws={'alpha':0.4}, line_kws={'lw':1})
                r, p = spearmanr(df[col], df["accident_count"], nan_policy='omit')
                ax.set_title(f"{col} vs accident_count  (Spearman ρ={r:.2f})")
                plt.tight_layout()
                st.pyplot(fig)
        with col2:
            pass

tabs = st.tabs(["데이터셋 구성","송파(예시) - 미구현"])
with tabs[0]:
    initialize_session_state()
    # --- 단계별 UI 렌더링 ---
    if st.session_state.tab0_current_step >= 0:
        # 0단계 UI는 항상 표시
        display_step0_region_selection()

    # 0단계 지역 선택이 완료된 후
    if st.session_state.tab0_current_step >= 1:
        st.divider() # 구분선
        # 1단계 UI 표시
        display_step1_sports_selection()

    # 1단계 스포츠 지역 선택이 완료된 후
    if st.session_state.tab0_current_step >= 2:
        st.divider()
        # 2단계 UI 표시
        display_step2_weekday_selection()
    
    # 2단계 요일 선택이 완료된 후
    if st.session_state.tab0_current_step >= 3:
        st.divider()
        # 3단계 UI 표시
        display_step3_dates_selection()

    # 3단계 날짜 선택이 완료된 후
    if st.session_state.tab0_current_step >= 4:
        st.divider()   
        # 4단계 UI 표시
        display_step4_build_dataset()

    # 4단계 데이터셋 빌드 완료된 후
    if st.session_state.tab0_current_step >= 5:
        st.divider() 
        # 5단계 UI 표시
        display_step5_dataset_info()
    
    # 5단계 데이터셋 정보 표시 후
    if st.session_state.tab0_current_step >= 6:
        st.divider()  
        # 6단계 UI 표시
        display_step6_dataset_heatmap()
        
    # 6단계 데이터셋 히트맵 표시 후
    if st.session_state.tab0_current_step >= 7:
        st.divider()   
        # 6단계 UI 표시
        display_step7_dataset_linear()
        
    # 데이터셋 다시 생성 버튼 (항상 제일 하단에 표시)
    if st.session_state.tab0_current_step >= 4:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.divider()
        # 데이터셋 리셋 버튼
        display_reset_dataset()

with tabs[1]:
    # 세션 상태 초기화
    if 'tab1_current_step' not in st.session_state:
        st.session_state.tab1_current_step = 0