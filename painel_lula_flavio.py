import os
import re
import unicodedata

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from wordcloud import WordCloud
from collections import Counter

# =========================================================
# CONFIGURAÇÃO
# =========================================================
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("Defina YOUTUBE_API_KEY no arquivo .env")

BASE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
BASE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
BASE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

st.set_page_config(
    page_title="Eleições Presidenciais 2026",
    page_icon="🗳️",
    layout="wide"
)

# =========================================================
# ELEIÇÕES PRESIDENCIAIS 2026
# =========================================================
CANDIDATOS = {}


def adicionar_candidato(nome, partido, termos_obrigatorios):
    """Mantém a mesma regra de coleta para todos os nomes monitorados."""
    nome_painel = f"{nome} ({partido})"
    CANDIDATOS[nome_painel] = {
        "queries": [
            f'"{nome}" eleições presidenciais 2026',
            f'"{nome}" candidato presidente 2026',
            f'"{nome}" pesquisa eleitoral presidente',
            f'"{nome}" campanha presidencial',
            f'"{nome}" política Brasil'
        ],
        "required_any": termos_obrigatorios,
        "accepted_any": [
            "presidente", "presidencia", "presidência", "presidencial",
            "eleicao", "eleição", "eleicoes", "eleições", "2026",
            "pre candidatura", "pré candidatura", "candidato", "candidatura",
            "campanha", "pesquisa", "politica", "política", "brasil"
        ],
        "blocked_any": [
            "musica", "música", "clipe", "filme", "gameplay", "minecraft",
            "futebol", "novela", "receita"
        ]
    }


adicionar_candidato(
    "Luiz Inácio Lula da Silva", "PT",
    [
        "luiz inacio lula da silva", "luiz inácio lula da silva",
        "presidente lula", "lula"
    ]
)
adicionar_candidato(
    "Flávio Bolsonaro", "PL",
    ["flavio bolsonaro", "flávio bolsonaro"]
)

STOPWORDS_PT = {
    "a", "o", "e", "da", "de", "do", "das", "dos",
    "um", "uma", "uns", "umas", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sobre", "entre", "contra",
    "eu", "tu", "ele", "ela", "eles", "elas", "nós", "nos",
    "voces", "vocês", "me", "te", "lhe", "lhes", "meu", "minha",
    "seu", "sua", "dele", "dela", "que", "como", "porque", "pq",
    "quando", "onde", "mais", "menos", "muito", "muita", "muitos",
    "muitas", "tambem", "também", "isso", "isto", "esse", "essa",
    "aquele", "aquela", "sim", "nao", "não", "ja", "já", "vou",
    "vai", "vc", "vcs", "ta", "tá", "to", "tô",
    "youtube", "video", "vídeo", "canal", "comentario", "comentário",
    "link", "live", "podcast", "stream",
    "governo", "governador", "campanha", "candidato", "candidata",
    "eleicao", "eleições", "politica", "política", "pesquisa",
    "debate", "voto", "votar", "prefeito", "presidente", "senador",
    "deputado", "brasil", "brasileiro", "brasileira",
    "luiz", "inacio", "inácio", "lula", "silva", "flavio", "flávio",
    "bolsonaro", "romeu", "zema", "ronaldo", "caiado", "renan", "santos",
    "augusto", "cury", "cabo", "daciolo", "clariana", "barao", "barão",
    "hertz", "dias", "edmilson", "costa", "samara", "martins", "rui",
    "pimenta", "leonardo", "avalanche",
    "2024", "2025", "2026",
    "bom", "boa", "otimo", "ótimo", "ruim", "melhor", "pior",
    "grande", "pequeno", "verdade", "verdadeiro", "falso", "fake"
}

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================


def normalizar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"http\S+", " ", texto)
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def contem_qualquer(texto: str, lista_termos: list[str]) -> bool:
    texto_norm = normalizar_texto(texto)
    return any(normalizar_texto(termo) in texto_norm for termo in lista_termos)


def score_texto(texto: str, termos: list[str]) -> int:
    texto_norm = normalizar_texto(texto)
    return sum(1 for termo in termos if normalizar_texto(termo) in texto_norm)


def percentual(valor, total):
    """Calcula percentual para números simples ou colunas pandas."""
    if isinstance(total, (pd.Series, pd.DataFrame)):
        return (valor / total.replace(0, pd.NA)) * 100

    if total == 0:
        return 0

    return (valor / total) * 100


def classificar_sentimento(texto: str) -> str:
    texto = normalizar_texto(texto)

    negativos = [
        "corrupto", "corrupta", "ladr", "roubo", "fraude", "golpe",
        "vergonha", "absurdo", "absurda", "desastre", "pessimo",
        "péssimo", "revoltante", "revoltado", "revoltada",
        "mentiroso", "mentirosa", "incompetente", "indignado",
        "indignada", "crime", "triste", "odio", "ódio"
    ]

    positivos = [
        "bom", "boa", "otimo", "ótimo", "otima", "ótima",
        "excelente", "parabens", "parabéns", "justo", "justa",
        "correto", "correta", "apoio", "confio", "melhor",
        "preparado", "preparada", "capaz"
    ]

    if any(p in texto for p in negativos):
        return "negativo"

    if any(p in texto for p in positivos):
        return "positivo"

    return "neutro"


def score_sentimento(sentimento: str) -> int:
    """Converte sentimento em pontuação para o termômetro."""
    if sentimento == "positivo":
        return 1
    if sentimento == "negativo":
        return -1
    return 0


def gerar_nuvem_palavras_comentarios(df_comentarios: pd.DataFrame):
    if df_comentarios.empty or "comentario" not in df_comentarios.columns:
        return None

    textos_comentarios = " ".join(
        df_comentarios["comentario"].dropna().astype(str).tolist()
    )

    nomes_candidatos = " ".join(
        df_comentarios["candidato"].dropna().astype(str).tolist()
    )

    textos = textos_comentarios + " " + nomes_candidatos
    textos = normalizar_texto(textos)

    stopwords_nuvem = STOPWORDS_PT.copy()

    nomes_para_manter = {
        "lula", "flavio", "bolsonaro", "zema", "caiado", "renan",
        "cury", "daciolo", "clariana", "hertz", "edmilson", "samara",
        "pimenta", "avalanche"
    }

    stopwords_nuvem = stopwords_nuvem - nomes_para_manter

    palavras = [
        p for p in textos.split()
        if len(p) >= 3
        and p not in stopwords_nuvem
        and not p.isdigit()
    ]

    texto_final = " ".join(palavras).strip()

    if not texto_final:
        return None

    wc = WordCloud(
        width=1400,
        height=650,
        background_color="white",
        colormap="Oranges",
        collocations=False
    ).generate(texto_final)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    return fig


# =========================================================
# API YOUTUBE
# =========================================================
def buscar_videos_por_query(
    query,
    max_results=25,
    region_code="BR",
    relevance_language="pt",
    order="relevance"
):
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "regionCode": region_code,
        "relevanceLanguage": relevance_language,
        "order": order,
        "key": API_KEY
    }

    resp = requests.get(BASE_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    return resp.json()


def buscar_detalhes_videos(video_ids):
    if not video_ids:
        return pd.DataFrame()

    params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": API_KEY
    }

    resp = requests.get(BASE_VIDEOS_URL, params=params, timeout=30)
    resp.raise_for_status()

    registros = []

    for item in resp.json().get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})

        registros.append({
            "video_id": item.get("id", ""),
            "titulo": snippet.get("title", ""),
            "descricao": snippet.get("description", ""),
            "canal": snippet.get("channelTitle", ""),
            "publicado_em": snippet.get("publishedAt", ""),
            "views": int(stats.get("viewCount", 0)) if "viewCount" in stats else 0,
            "likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
            "comentarios": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0,
            "url": f"https://www.youtube.com/watch?v={item.get('id', '')}"
        })

    df = pd.DataFrame(registros)

    if not df.empty:
        df["publicado_em"] = pd.to_datetime(
            df["publicado_em"], errors="coerce", utc=True)

    return df


def buscar_comentarios(video_id, max_results=50):
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "order": "relevance",
        "textFormat": "plainText",
        "key": API_KEY
    }

    try:
        resp = requests.get(BASE_COMMENTS_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    comentarios = []

    for item in data.get("items", []):
        try:
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comentarios.append({
                "video_id": video_id,
                "comentario": snippet.get("textDisplay", ""),
                "autor_comentario": snippet.get("authorDisplayName", ""),
                "like_comentario": int(snippet.get("likeCount", 0)),
                "publicado_em_comentario": snippet.get("publishedAt", "")
            })
        except Exception:
            continue

    return comentarios


# =========================================================
# FILTRO
# =========================================================
def validar_video_candidato(texto_base: str, regra: dict) -> tuple[bool, int]:
    if contem_qualquer(texto_base, regra["blocked_any"]):
        return False, 0

    tem_required = contem_qualquer(texto_base, regra["required_any"])
    tem_accepted = contem_qualquer(texto_base, regra["accepted_any"])

    if not tem_required or not tem_accepted:
        return False, 0

    score = (
        score_texto(texto_base, regra["required_any"]) * 3 +
        score_texto(texto_base, regra["accepted_any"]) * 2
    )

    return True, score


def coletar_videos_candidato(
    nome_candidato,
    regra_candidato,
    max_results_por_busca=20,
    region_code="BR",
    relevance_language="pt",
    order="relevance"
):
    todos = []

    for consulta in regra_candidato["queries"]:
        try:
            data = buscar_videos_por_query(
                query=consulta,
                max_results=max_results_por_busca,
                region_code=region_code,
                relevance_language=relevance_language,
                order=order
            )

            video_ids = [
                item.get("id", {}).get("videoId")
                for item in data.get("items", [])
                if item.get("id", {}).get("videoId")
            ]

            df_det = buscar_detalhes_videos(video_ids)

            if df_det.empty:
                continue

            df_det["texto_base"] = (
                df_det["titulo"].fillna("") + " " +
                df_det["descricao"].fillna("")
            )

            df_det["candidato"] = nome_candidato
            df_det["consulta_usada"] = consulta

            flags = df_det["texto_base"].apply(
                lambda x: validar_video_candidato(x, regra_candidato)
            )

            df_det["aprovado"] = flags.apply(lambda x: x[0])
            df_det["score_candidato"] = flags.apply(lambda x: x[1])
            df_det = df_det[df_det["aprovado"]].copy()

            if not df_det.empty:
                todos.append(df_det)

        except Exception as e:
            print(f"[ERRO] Consulta '{consulta}' falhou: {e}")

    if not todos:
        return pd.DataFrame()

    df_final = pd.concat(todos, ignore_index=True)

    df_final = (
        df_final.sort_values(
            by=["views", "score_candidato", "likes", "comentarios"],
            ascending=[False, False, False, False]
        )
        .drop_duplicates(subset=["video_id"], keep="first")
        .reset_index(drop=True)
    )

    return df_final


# =========================================================
# CARGA DE DADOS
# =========================================================
@st.cache_data(ttl=1800)
def carregar_dados(
    max_results_por_busca=20,
    region_code="BR",
    relevance_language="pt",
    order="relevance"
):
    frames = []

    for nome_candidato, regra in CANDIDATOS.items():
        df = coletar_videos_candidato(
            nome_candidato=nome_candidato,
            regra_candidato=regra,
            max_results_por_busca=max_results_por_busca,
            region_code=region_code,
            relevance_language=relevance_language,
            order=order
        )

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_videos = pd.concat(frames, ignore_index=True)

    ranking_candidato = (
        df_videos.groupby("candidato", as_index=False)
        .agg(
            quantidade_videos=("video_id", "count"),
            total_views=("views", "sum"),
            total_likes=("likes", "sum"),
            total_comentarios=("comentarios", "sum")
        )
        .sort_values(by=["total_views", "quantidade_videos"], ascending=[False, False])
        .reset_index(drop=True)
    )

    ranking_canais = (
        df_videos.groupby("canal", as_index=False)
        .agg(
            quantidade_videos=("video_id", "count"),
            total_views=("views", "sum")
        )
        .sort_values(by=["total_views", "quantidade_videos"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return df_videos, ranking_candidato, ranking_canais


# =========================================================
# COMENTÁRIOS POR CANDIDATO
# =========================================================
@st.cache_data(ttl=1800)
def carregar_comentarios(
    df_videos,
    max_videos_por_candidato=5,
    max_comments_per_video=20
):
    if df_videos.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    comentarios_lista = []
    candidatos_unicos = df_videos["candidato"].unique()

    for candidato in candidatos_unicos:
        df_candidato = (
            df_videos[df_videos["candidato"] == candidato]
            .sort_values(
                by=["views", "likes", "comentarios"],
                ascending=[False, False, False]
            )
            .head(max_videos_por_candidato)
        )

        if df_candidato.empty:
            continue

        mapa_video_titulo = dict(
            zip(df_candidato["video_id"], df_candidato["titulo"]))
        mapa_video_canal = dict(
            zip(df_candidato["video_id"], df_candidato["canal"]))

        for video_id in df_candidato["video_id"]:
            comentarios = buscar_comentarios(
                video_id,
                max_results=max_comments_per_video
            )

            for c in comentarios:
                texto = c["comentario"]

                comentarios_lista.append({
                    "video_id": video_id,
                    "candidato": candidato,
                    "titulo_video": mapa_video_titulo.get(video_id, ""),
                    "canal": mapa_video_canal.get(video_id, ""),
                    "comentario": texto,
                    "sentimento": classificar_sentimento(texto),
                    "autor_comentario": c["autor_comentario"],
                    "like_comentario": c["like_comentario"],
                    "publicado_em_comentario": c["publicado_em_comentario"]
                })

    df_comentarios = pd.DataFrame(comentarios_lista)

    if df_comentarios.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ranking_sentimento = (
        df_comentarios.groupby("sentimento", as_index=False)
        .agg(quantidade=("comentario", "count"))
        .sort_values(by="quantidade", ascending=False)
        .reset_index(drop=True)
    )

    ranking_sentimento_candidato = (
        df_comentarios.groupby(["candidato", "sentimento"], as_index=False)
        .agg(quantidade=("comentario", "count"))
    )

    sentimentos = ["positivo", "negativo", "neutro"]
    linhas_completas = []

    for cand in candidatos_unicos:
        for sent in sentimentos:
            filtro = ranking_sentimento_candidato[
                (ranking_sentimento_candidato["candidato"] == cand) &
                (ranking_sentimento_candidato["sentimento"] == sent)
            ]

            if filtro.empty:
                linhas_completas.append({
                    "candidato": cand,
                    "sentimento": sent,
                    "quantidade": 0
                })

    if linhas_completas:
        ranking_sentimento_candidato = pd.concat(
            [
                ranking_sentimento_candidato,
                pd.DataFrame(linhas_completas)
            ],
            ignore_index=True
        )

    # PERCENTUAL POR LINHA:
    # positivo soma 100%, neutro soma 100%, negativo soma 100%
    ranking_sentimento_candidato["percentual"] = (
        ranking_sentimento_candidato.groupby("sentimento")["quantidade"]
        .transform(
            lambda x: (
                (x / x.sum()) * 100
                if x.sum() > 0
                else 0
            )
        )
    )

    return df_comentarios, ranking_sentimento, ranking_sentimento_candidato


# =========================================================
# DASHBOARD
# =========================================================
st.title("🗳️ Painel Nacional — Lula x Flávio Bolsonaro")
st.markdown("""
Monitoramento de vídeos e comentários públicos do YouTube com escopo nacional (Brasil).

Comparação exclusiva: **Luiz Inácio Lula da Silva (PT) × Flávio Bolsonaro (PL)**.

> Os resultados refletem os conteúdos retornados pela API do YouTube, não uma pesquisa eleitoral
> nem uma medição representativa da opinião de todos os brasileiros.
""")

with st.sidebar:
    st.header("Filtros")

    region_code = "BR"  # Busca nacional no YouTube, em todo o Brasil
    relevance_language = "pt"  # Conteúdo em português
    order = st.selectbox("Ordenação da busca", ["relevance", "date"], index=0)
    max_results_por_busca = st.slider("Vídeos por consulta", 5, 50, 30, 5)

    st.markdown("---")
    st.subheader("Comentários por candidato")

    max_videos_comentarios = st.slider("Vídeos por candidato", 1, 10, 5)
    max_comments_per_video = st.slider("Comentários por vídeo", 5, 50, 20)
    st.markdown("---")
    st.subheader("⚔️ Comparação: Lula x Flávio Bolsonaro")

    candidato_a = st.selectbox(
        "Candidato A",
        list(CANDIDATOS.keys()),
        index=0
    )

    candidato_b = st.selectbox(
        "Candidato B",
        list(CANDIDATOS.keys()),
        index=1
    )

    atualizar = st.button("Atualizar dados")

if atualizar:
    st.cache_data.clear()

try:
    df_videos, ranking_candidato, ranking_canais = carregar_dados(
        max_results_por_busca=max_results_por_busca,
        region_code=region_code,
        relevance_language=relevance_language,
        order=order
    )
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

if df_videos.empty:
    st.warning(
        "Nenhum vídeo encontrado para Lula ou Flávio Bolsonaro nas consultas nacionais.")
    st.stop()

# Filtro temporal opcional sobre a data de publicação dos vídeos.
with st.sidebar:
    periodo_dias = st.selectbox(
        "Período dos vídeos",
        options=[7, 30, 90, 365],
        index=1,
        format_func=lambda d: f"Últimos {d} dias"
    )

if "publicado_em" in df_videos.columns:
    datas_publicacao = pd.to_datetime(
        df_videos["publicado_em"], errors="coerce", utc=True)
    limite_data = pd.Timestamp.now(
        tz="UTC") - pd.Timedelta(days=int(periodo_dias))
    df_videos = df_videos[datas_publicacao >= limite_data].copy()
    if df_videos.empty:
        st.warning(
            f"Nenhum vídeo encontrado nos últimos {periodo_dias} dias. Tente ampliar o período.")
        st.stop()

with st.spinner("Lendo comentários por candidato..."):
    df_comentarios, ranking_sentimento, ranking_sentimento_candidato = carregar_comentarios(
        df_videos=df_videos,
        max_videos_por_candidato=max_videos_comentarios,
        max_comments_per_video=max_comments_per_video
    )
    df_comparacao = df_comentarios[
        df_comentarios["candidato"].isin(
            [candidato_a, candidato_b]
        )
    ].copy()

# =========================================================
# KPIs
# =========================================================
total_views = df_videos["views"].sum()
total_likes = df_videos["likes"].sum()
total_comentarios = df_videos["comentarios"].sum()

perc_likes = percentual(total_likes, total_views)
perc_comentarios = percentual(total_comentarios, total_views)
perc_engajamento = percentual(total_likes + total_comentarios, total_views)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total de Vídeos", f"{len(df_videos):,}".replace(",", "."))

c2.metric("Total de Curtidas", f"{int(total_likes):,}".replace(",", "."))

c3.metric("Total de Comentários",
          f"{int(total_comentarios):,}".replace(",", "."))

c4.metric("Total de Engajamento",
          f"{int(total_likes + total_comentarios):,}".replace(",", "."))

# =========================================================
# PERCENTUAIS
# =========================================================
ranking_candidato["perc_views"] = percentual(
    ranking_candidato["total_views"],
    ranking_candidato["total_views"].sum()
)

ranking_candidato["perc_likes"] = percentual(
    ranking_candidato["total_likes"],
    ranking_candidato["total_likes"].sum()
)

ranking_candidato["perc_comentarios"] = percentual(
    ranking_candidato["total_comentarios"],
    ranking_candidato["total_comentarios"].sum()
)

ranking_canais["perc_views"] = percentual(
    ranking_canais["total_views"],
    ranking_canais["total_views"].sum()
)

# =========================================================
# GRÁFICOS
# =========================================================
g1, g2 = st.columns(2)

with g1:
    st.subheader("📊 Percentual de Views por Candidato")

    fig1 = px.bar(
        ranking_candidato,
        x="candidato",
        y="perc_views",
        text=ranking_candidato["perc_views"].round(1).astype(str) + "%",
        color="candidato"
    )

    fig1.update_layout(
        xaxis_title="Candidato",
        yaxis_title="Percentual (%)"
    )

    st.plotly_chart(fig1, use_container_width=True)

with g2:
    st.subheader("📺 Percentual de Views por Canal")

    top_canais = ranking_canais.head(15).copy()

    fig2 = px.bar(
        top_canais,
        x="perc_views",
        y="canal",
        orientation="h",
        text=top_canais["perc_views"].round(1).astype(str) + "%"
    )

    fig2.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Percentual (%)",
        yaxis_title="Canal"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# MAPA DE CALOR COMENTÁRIOS %
# =========================================================
st.subheader("🔥 Percentual de Comentários por Candidato")

heatmap_df = ranking_candidato[["candidato", "perc_comentarios"]].copy()

if not heatmap_df.empty:
    matriz_heatmap = pd.DataFrame(
        [heatmap_df["perc_comentarios"].tolist()],
        columns=heatmap_df["candidato"].tolist(),
        index=["Comentários (%)"]
    )

    fig3 = px.imshow(
        matriz_heatmap,
        text_auto=".1f",
        aspect="auto",
        labels=dict(x="Candidato", y="", color="%"),
        color_continuous_scale="Oranges"
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# SENTIMENTO %
# =========================================================
st.subheader(
    "😊 Distribuição percentual dos candidatos dentro de cada sentimento")

if not df_comentarios.empty:
    ranking_sentimento["percentual"] = percentual(
        ranking_sentimento["quantidade"],
        ranking_sentimento["quantidade"].sum()
    )

    c5, c6 = st.columns(2)

    with c5:
        fig4 = px.bar(
            ranking_sentimento,
            x="sentimento",
            y="percentual",
            text=ranking_sentimento["percentual"].round(1).astype(str) + "%",
            color="sentimento",
            color_discrete_map={
                "positivo": "green",
                "negativo": "red",
                "neutro": "lightgray"
            }
        )

        fig4.update_layout(
            xaxis_title="Sentimento",
            yaxis_title="Percentual (%)",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig4, use_container_width=True)

    with c6:
        pivot_sentimento = (
            ranking_sentimento_candidato
            .pivot_table(
                index="sentimento",
                columns="candidato",
                values="percentual",
                aggfunc="sum"
            )
            .fillna(0)
        )

        pivot_sentimento = pivot_sentimento.reindex(
            ["positivo", "neutro", "negativo"]
        )

        fig5 = px.imshow(
            pivot_sentimento,
            text_auto=".1f",
            aspect="auto",
            labels=dict(
                x="Candidato",
                y="Sentimento",
                color="%"
            ),
            color_continuous_scale="Oranges"
        )

        fig5.update_layout(
            title="Percentual por linha: cada sentimento soma 100%",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig5, use_container_width=True)

    # =========================================================
    # TERMÔMETRO DE SENTIMENTO DOS CANDIDATOS
    # =========================================================
    st.subheader("🌡️ Termômetro médio de sentimento por candidato")

    df_termometro = df_comentarios.copy()
    df_termometro["score_sentimento"] = df_termometro["sentimento"].apply(
        score_sentimento)

    termometro = (
        df_termometro
        .groupby("candidato", as_index=False)
        .agg(
            indice_medio=("score_sentimento", "mean"),
            comentarios_analisados=("comentario", "count")
        )
    )

    termometro["termometro"] = (termometro["indice_medio"] * 100).round(1)
    media_geral = termometro["termometro"].mean().round(1)

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        st.metric("Termômetro médio geral", f"{media_geral}%")

        if media_geral >= 25:
            st.success("Percepção geral positiva")
        elif media_geral <= -25:
            st.error("Percepção geral negativa")
        else:
            st.warning("Percepção geral neutra/dividida")

    with col_t2:
        termometro_plot = termometro.sort_values(
            "termometro", ascending=True).copy()

        fig_termometro = px.bar(
            termometro_plot,
            x="termometro",
            y="candidato",
            orientation="h",
            text=termometro_plot["termometro"].astype(str) + "%",
            color="termometro",
            color_continuous_scale=["red", "lightgray", "green"],
            range_color=[-100, 100],
            title="Termômetro de sentimento por candidato"
        )

        fig_termometro.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color="black"
        )

        fig_termometro.update_layout(
            xaxis_title="Sentimento médio (-100 a +100)",
            yaxis_title="Candidato",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig_termometro, use_container_width=True)

    st.dataframe(
        termometro.sort_values("termometro", ascending=False),
        use_container_width=True
    )

    st.subheader("☁️ Nuvem de Palavras")

    fig_wc = gerar_nuvem_palavras_comentarios(df_comentarios)

    if fig_wc is not None:
        st.pyplot(fig_wc, use_container_width=True)
    else:
        st.info("Sem palavras suficientes para gerar a nuvem.")

    st.subheader("📋 Comentários Coletados")

    st.dataframe(
        df_comentarios[
            [
                "candidato",
                "titulo_video",
                "canal",
                "comentario",
                "sentimento",
                "autor_comentario",
                "like_comentario"
            ]
        ],
        use_container_width=True,
        height=500
    )

else:
    st.warning("Não foi possível carregar comentários.")
# =========================================================
# COMPARAÇÃO ENTRE CANDIDATOS
# =========================================================

st.header("⚔️ Comparação Estratégica")

if not df_comparacao.empty:

    # --------------------------------------------------
    # SHARE OF VOICE
    # --------------------------------------------------
    st.subheader("📢 Share of Voice")

    share = (
        df_comparacao
        .groupby("candidato", as_index=False)
        .agg(
            mencoes=("comentario", "count")
        )
    )

    share["share"] = (
        share["mencoes"]
        /
        share["mencoes"].sum()
    ) * 100

    fig_share = px.pie(
        share,
        names="candidato",
        values="share",
        title="Participação nas menções"
    )

    st.plotly_chart(
        fig_share,
        use_container_width=True
    )

    # --------------------------------------------------
    # SENTIMENTO LÍQUIDO
    # --------------------------------------------------
    st.subheader("🌡️ Sentimento Líquido")

    sentimento_liquido = (
        df_comparacao
        .groupby(
            ["candidato", "sentimento"]
        )
        .size()
        .unstack(fill_value=0)
    )

    sentimento_liquido["sentimento_liquido"] = (
        (
            sentimento_liquido.get("positivo", 0)
            -
            sentimento_liquido.get("negativo", 0)
        )
        /
        sentimento_liquido.sum(axis=1)
    ) * 100

    st.dataframe(
        sentimento_liquido[
            ["sentimento_liquido"]
        ]
        .round(2)
    )

    # --------------------------------------------------
    # EVOLUÇÃO TEMPORAL
    # --------------------------------------------------
    st.subheader("📈 Evolução das Menções")

    temp = df_comparacao.copy()
##########################
# Correção fuso horário
###########################
    temp["data"] = (
        pd.to_datetime(
            temp["publicado_em_comentario"],
            errors="coerce",
            utc=True
        )
        .dt.date
    )

    limite = (
        pd.Timestamp.now()
        - pd.Timedelta(days=periodo_dias)
    ).date()

    temp = temp[
        temp["data"] >= limite
    ]
#############################
# Fim
#############################
    evolucao = (
        temp
        .groupby(
            [
                "data",
                "candidato"
            ]
        )
        .size()
        .reset_index(name="comentarios")
    )
    st.write("Candidatos na evolução:")
    st.write(evolucao["candidato"].value_counts())

    st.write(evolucao.head(20))

    fig_evolucao = px.line(
        evolucao,
        x="data",
        y="comentarios",
        color="candidato",
        markers=True
    )

    st.plotly_chart(
        fig_evolucao,
        use_container_width=True
    )

    # --------------------------------------------------
    # NARRATIVAS POSITIVAS
    # --------------------------------------------------
    st.subheader("✅ Narrativas Positivas")

    texto_pos = " ".join(
        df_comparacao[
            df_comparacao["sentimento"] == "positivo"
        ]["comentario"].astype(str)
    )

    palavras_pos = [
        p
        for p in normalizar_texto(
            texto_pos
        ).split()
        if len(p) > 3
        and p not in STOPWORDS_PT
    ]

    top_pos = pd.DataFrame(
        Counter(
            palavras_pos
        ).most_common(20),
        columns=["Palavra", "Frequência"]
    )

    st.dataframe(
        top_pos,
        use_container_width=True
    )

    # --------------------------------------------------
    # NARRATIVAS NEGATIVAS
    # --------------------------------------------------
    st.subheader("❌ Narrativas Negativas")

    texto_neg = " ".join(
        df_comparacao[
            df_comparacao["sentimento"] == "negativo"
        ]["comentario"].astype(str)
    )

    palavras_neg = [
        p
        for p in normalizar_texto(
            texto_neg
        ).split()
        if len(p) > 3
        and p not in STOPWORDS_PT
    ]

    top_neg = pd.DataFrame(
        Counter(
            palavras_neg
        ).most_common(20),
        columns=["Palavra", "Frequência"]
    )

    st.dataframe(
        top_neg,
        use_container_width=True
    )
############################################################
# Inserção do Novo Módulo
##########################################################
    # ==================================================
# INTELIGÊNCIA DE CANAIS
# ==================================================

    st.subheader("📺 Comparativo de Canais")

    canais_comp = (
        df_videos[
            df_videos["candidato"].isin(
                [
                    candidato_a,
                    candidato_b
                ]
            )
        ]
        .groupby(
            ["canal", "candidato"],
            as_index=False
        )
        .agg(
            videos=("video_id", "count"),
            views=("views", "sum"),
            likes=("likes", "sum"),
            comentarios=("comentarios", "sum")
        )
    )
    fig_canais = px.bar(
        canais_comp,
        x="canal",
        y="views",
        color="candidato",
        barmode="group",
        text="views",
        title="Views por canal"
    )

    fig_canais.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Canal",
        yaxis_title="Views"
    )

    st.plotly_chart(
        fig_canais,
        use_container_width=True
    )
    st.subheader("🏆 Ranking de Canais")

    ranking_canais_comp = (
        canais_comp
        .sort_values(
            "views",
            ascending=False
        )
    )

    st.dataframe(
        ranking_canais_comp[
            [
                "canal",
                "candidato",
                "videos",
                "views",
                "likes",
                "comentarios"
            ]
        ],
        use_container_width=True
    )
    st.subheader("🤝 Canais em Comum")

canais_a = set(
    df_videos[
        df_videos["candidato"] == candidato_a
    ]["canal"]
)

canais_b = set(
    df_videos[
        df_videos["candidato"] == candidato_b
    ]["canal"]
)

canais_comum = sorted(
    list(
        canais_a.intersection(canais_b)
    )
)

st.metric(
    "Quantidade",
    len(canais_comum)
)

if canais_comum:

    st.dataframe(
        pd.DataFrame(
            canais_comum,
            columns=["Canal"]
        ),
        use_container_width=True
    )
else:
    st.info("Nenhum canal em comum.")
col1, col2 = st.columns(2)

with col1:

    exclusivos_a = sorted(
        list(
            canais_a - canais_b
        )
    )

    st.subheader(
        f"📢 Exclusivos de {candidato_a}"
    )

    st.dataframe(
        pd.DataFrame(
            exclusivos_a,
            columns=["Canal"]
        ),
        height=300,
        use_container_width=True
    )

with col2:

    exclusivos_b = sorted(
        list(
            canais_b - canais_a
        )
    )

    st.subheader(
        f"📢 Exclusivos de {candidato_b}"
    )

    st.dataframe(
        pd.DataFrame(
            exclusivos_b,
            columns=["Canal"]
        ),
        height=300,
        use_container_width=True
    )
    st.subheader("🚀 Canal Mais Influente")

    top_canal = (
        canais_comp
        .sort_values(
            "views",
            ascending=False
        )
        .groupby("candidato")
        .head(1)
    )

    st.dataframe(
        top_canal[
            [
                "candidato",
                "canal",
                "views",
                "likes",
                "comentarios"
            ]
        ],
        use_container_width=True
    )
    st.subheader("🔥 Heatmap Canal × Candidato")

    heat = (
        canais_comp
        .pivot_table(
            index="canal",
            columns="candidato",
            values="views",
            aggfunc="sum"
        )
        .fillna(0)
    )

    fig_heat = px.imshow(
        heat,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True
    )

    # ==================================================
# INTELIGÊNCIA DE CANAIS
# ==================================================

    # --------------------------------------------------
    # TOP NOTÍCIAS
    # --------------------------------------------------
    st.subheader("📰 Notícias de Maior Repercussão")

    noticias = (
        df_videos[
            df_videos["candidato"].isin(
                [
                    candidato_a,
                    candidato_b
                ]
            )
        ]
        .sort_values(
            "views",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        noticias[
            [
                "candidato",
                "titulo",
                "views",
                "likes",
                "comentarios",
                "url"
            ]
        ],
        use_container_width=True
    )

    # --------------------------------------------------
    # COMENTÁRIOS MAIS RELEVANTES
    # --------------------------------------------------
    st.subheader("💬 Comentários Mais Relevantes")

    comentarios_relevantes = (
        df_comparacao
        .sort_values(
            "like_comentario",
            ascending=False
        )
        .head(30)
    )

    st.dataframe(
        comentarios_relevantes[
            [
                "candidato",
                "comentario",
                "sentimento",
                "like_comentario"
            ]
        ],
        use_container_width=True
    )

    # --------------------------------------------------
    # RANKING DIGITAL
    # --------------------------------------------------
    st.subheader("🏆 Índice de Popularidade Digital")

    ipd = (
        df_videos[
            df_videos["candidato"].isin(
                [
                    candidato_a,
                    candidato_b
                ]
            )
        ]
        .groupby(
            "candidato",
            as_index=False
        )
        .agg(
            views=("views", "sum"),
            likes=("likes", "sum"),
            comentarios=("comentarios", "sum")
        )
    )

    ipd["ipd"] = (
        ipd["views"] * 0.20
        +
        ipd["likes"] * 0.40
        +
        ipd["comentarios"] * 0.40
    )

    fig_ipd = px.bar(
        ipd,
        x="candidato",
        y="ipd",
        color="ipd",
        text="ipd"
    )

    st.plotly_chart(
        fig_ipd,
        use_container_width=True
    )

    # --------------------------------------------------
    # CONCLUSÃO ESTRATÉGICA
    # --------------------------------------------------
    st.subheader("🤖 Conclusão Estratégica")

    lider_ipd = (
        ipd
        .sort_values(
            "ipd",
            ascending=False
        )
        .iloc[0]["candidato"]
    )

    lider_sentimento = (
        sentimento_liquido
        .sort_values(
            "sentimento_liquido",
            ascending=False
        )
        .index[0]
    )

    st.success(f"""
📊 O candidato com maior força digital é: {lider_ipd}

😊 O candidato com melhor sentimento líquido é: {lider_sentimento}

✅ Analise Share of Voice, Sentimento Líquido e Evolução Temporal
para identificar quem está ampliando influência digital
e quais narrativas estão impulsionando ou prejudicando
cada candidatura.
""")
# =========================================================
# TABELA FINAL
# =========================================================
st.subheader("📺 Vídeos Aprovados em Percentual")

df_exibir = df_videos[
    [
        "candidato",
        "titulo",
        "canal",
        "views",
        "likes",
        "comentarios",
        "score_candidato",
        "consulta_usada",
        "publicado_em",
        "url"
    ]
].copy()

df_exibir["perc_views"] = percentual(
    df_exibir["views"], df_exibir["views"].sum())
df_exibir["perc_likes"] = percentual(
    df_exibir["likes"], df_exibir["likes"].sum())
df_exibir["perc_comentarios"] = percentual(
    df_exibir["comentarios"],
    df_exibir["comentarios"].sum()
)

df_exibir["publicado_em"] = df_exibir["publicado_em"].astype(str)

df_tabela_percentual = df_exibir[
    [
        "candidato",
        "titulo",
        "canal",
        "perc_views",
        "perc_likes",
        "perc_comentarios",
        "score_candidato",
        "consulta_usada",
        "publicado_em",
        "url"
    ]
].copy()

df_tabela_percentual["perc_views"] = df_tabela_percentual["perc_views"].round(
    2)
df_tabela_percentual["perc_likes"] = df_tabela_percentual["perc_likes"].round(
    2)
df_tabela_percentual["perc_comentarios"] = df_tabela_percentual["perc_comentarios"].round(
    2)

st.dataframe(
    df_tabela_percentual.sort_values(
        by="perc_views",
        ascending=False
    ),
    use_container_width=True,
    height=650
)

# =========================================================
# DOWNLOADS
# =========================================================
csv_videos = df_tabela_percentual.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "⬇️ Baixar CSV vídeos percentual",
    data=csv_videos,
    file_name="eleicoes_nacionais_videos_percentual.csv",
    mime="text/csv"
)

if not df_comentarios.empty:
    csv_comentarios = df_comentarios.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Baixar CSV comentários",
        data=csv_comentarios,
        file_name="eleicoes_nacionais_comentarios.csv",
        mime="text/csv"
    )
