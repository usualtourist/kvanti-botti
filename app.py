import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from tutor_bot import TutorBot, BotConfig, DATASET_EXERCISES


# --------------------------------------------------
# Perusasetukset
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Kvanti — Tilastotieteen tutori",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Teema
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --kvanti-bg: #f3f6fa;
        --kvanti-panel: #ffffff;
        --kvanti-panel-2: #eef3fb;
        --kvanti-text: #1f2933;
        --kvanti-muted: #5f6b76;
        --kvanti-primary: #2e86c1;
        --kvanti-primary-dark: #1b4f72;
        --kvanti-accent: #48c9b0;
        --kvanti-border: #cfd8e3;
        --kvanti-dark: #1c2833;
    }

    .stApp {
        background:
            linear-gradient(
                135deg,
                rgba(255, 255, 255, 0.90),
                rgba(238, 243, 251, 0.95)
            );
        color: var(--kvanti-text);
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(207, 216, 227, 0.9);
        border-radius: 14px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(31, 41, 51, 0.06);
    }

    p, li, span, label, div {
        color: var(--kvanti-text);
    }

    h1, h2, h3 {
        color: var(--kvanti-dark) !important;
        font-family: "Segoe UI", Arial, sans-serif;
        font-weight: 800;
    }

    h1 {
        border-bottom: 4px solid var(--kvanti-primary);
        padding-bottom: 0.45rem;
    }

    h2, h3 {
        border-left: 5px solid var(--kvanti-primary);
        padding-left: 0.55rem;
    }

    code {
        background-color: #eaf3fb;
        color: var(--kvanti-primary-dark);
        padding: 0.15rem 0.35rem;
        border-radius: 5px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbff 0%, #e6eef8 100%);
        border-right: 3px solid var(--kvanti-primary);
        min-width: 360px !important;
        width: 360px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        min-width: 360px !important;
        width: 360px !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--kvanti-text) !important;
    }

    .kvanti-banner {
        background: linear-gradient(90deg, #eaf3fb, #ffffff);
        border: 1px solid var(--kvanti-border);
        border-left: 7px solid var(--kvanti-primary);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        line-height: 1.55;
    }

    [data-testid="stChatMessage"] {
        background-color: var(--kvanti-panel);
        border: 1px solid var(--kvanti-border);
        border-left: 6px solid var(--kvanti-primary);
        border-radius: 12px;
        padding: 0.75rem;
        box-shadow: 0 4px 12px rgba(31, 41, 51, 0.06);
    }

    .stButton button,
    .stDownloadButton button {
        background: linear-gradient(180deg, #5dade2, #2e86c1) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 2px solid var(--kvanti-primary-dark) !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 0 var(--kvanti-primary-dark);
        letter-spacing: 0.2px;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        background: linear-gradient(180deg, #85c1e9, #3498db) !important;
        transform: translateY(-1px);
    }

    .stButton button:disabled,
    .stDownloadButton button:disabled {
        background: #d5dbdb !important;
        color: #6c7a89 !important;
        border-color: #aab7b8 !important;
        box-shadow: none;
    }

    [data-testid="stChatInput"] {
        background-color: #ffffff;
        border: 2px solid var(--kvanti-primary);
        border-radius: 12px;
    }

    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.75);
        border: 1px solid var(--kvanti-border);
        border-radius: 10px;
    }

    .right-panel {
        background: rgba(234, 243, 251, 0.95);
        border: 1px solid var(--kvanti-border);
        border-left: 6px solid var(--kvanti-primary);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .right-panel h3 {
        border-left: none !important;
        padding-left: 0 !important;
        margin-top: 0;
    }

    .right-panel-note {
        background: #ffffff;
        border: 1px solid var(--kvanti-border);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .learning-card {
        background-color: #ffffff;
        border: 1px solid var(--kvanti-border);
        border-left: 6px solid var(--kvanti-primary);
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.65rem;
    }

    .learning-card-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: var(--kvanti-dark);
        margin-bottom: 0.25rem;
    }

    .learning-card-status {
        color: var(--kvanti-text);
        margin-bottom: 0.25rem;
    }

    .learning-card-desc {
        color: var(--kvanti-muted);
        line-height: 1.45;
        font-size: 0.93rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API-avain
# --------------------------------------------------

def get_api_key() -> str | None:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if api_key:
            return api_key
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


# --------------------------------------------------
# Botin alustus
# --------------------------------------------------

def init_bot(study_topic: str) -> TutorBot:
    api_key = get_api_key()

    if not api_key:
        st.error(
            "📊 OPENAI_API_KEY puuttuu. Lisää se `.env`-tiedostoon "
            "tai Streamlitin secretsiin."
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    config = BotConfig(
        custom_topic=study_topic,
        main_model=st.session_state.get("main_model", "gpt-4o"),
        judge_model="gpt-4o-mini",
        temperature=st.session_state.get("temperature", 0.4),
        max_turns=st.session_state.get("max_turns", 40),
    )

    return TutorBot(client, config)


def reset_session(study_topic: str) -> None:
    """
    Aloittaa uuden opiskeluhetken käyttäjän itse syöttämällä aiheella.
    Kvanti vastaa heti ensimmäisen aiheen perusteella.
    """
    st.session_state.bot = init_bot(study_topic)
    st.session_state.messages = []
    st.session_state.study_summary = None
    st.session_state.review = None
    st.session_state.dataset_exercise = None
    st.session_state.study_topic = study_topic
    st.session_state.lesson_goal = study_topic
    st.session_state.study_started = True

    bot: TutorBot = st.session_state.bot

    # Näytetään opiskelijan syöttämä aihe keskustelussa käyttäjän viestinä.
    visible_user_msg = f"Tällä oppitunnilla haluan opiskella tätä: {study_topic}"

    st.session_state.messages.append(
        {"role": "user", "content": visible_user_msg}
    )

    # Lähetetään Kvantille sisäinen aloitusohje heti.
    first_prompt = (
        f"Opiskelijan opiskeltava aihe tällä oppitunnilla on: {study_topic}. "
        "Aloita keskustelu heti. "
        "Tee seuraavat asiat: "
        "1) Kuittaa opiskelijan aihe yhdellä lauseella. "
        "2) Kerro lyhyesti, mihin kvantitatiivisen tutkimuksen osa-alueeseen aihe liittyy. "
        "3) Anna yksi pieni ihmistieteellinen esimerkki. "
        "4) Kysy yksi tarkentava kysymys opiskelijalta. "
        "Älä jää odottamaan opiskelijan seuraavaa viestiä."
    )

    reply = bot.ask(first_prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )



# --------------------------------------------------
# Session state
# --------------------------------------------------

for key, default in [
    ("bot", None),
    ("messages", []),
    ("study_summary", None),
    ("review", None),
    ("dataset_exercise", None),
    ("study_topic", None),
    ("lesson_goal", ""),
    ("lesson_goal_input", ""),
    ("study_started", False),
    ("main_model", "gpt-4o"),
    ("temperature", 0.4),
    ("max_turns", 40),
    ("consent_given", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# Sivupalkki
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 Kvanti")

    st.warning(
        "🔒 **Tietoturva:** Viestit lähetetään OpenAI:lle käsiteltäväksi. "
        "Älä syötä henkilötietoja tai arkaluonteisia tietoja."
    )

    with st.expander("🔒 Lue lisää tietoturvasta", expanded=False):
        st.markdown(
            """
**Mihin viestit menevät?**

Kvanti käyttää OpenAI:n kielimallia. Kaikki kirjoittamasi viestit
lähetetään OpenAI:n palveluun käsiteltäväksi vastausta varten.

**OpenAI:n tietoturva- ja yksityisyysperiaatteet:**  
[https://openai.com/fi-FI/security-and-privacy/](https://openai.com/fi-FI/security-and-privacy/)

**Älä syötä:**

- henkilötietoja
- tunnistettavia tietoja muista henkilöistä
- terveystietoja tai muita arkaluonteisia tietoja
- salasanoja tai API-avaimia
- salassa pidettävää tutkimus- tai opiskelumateriaalia

Käytä mieluummin anonymisoituja tai keksittyjä esimerkkejä.
            """
        )

    st.divider()

    with st.expander("📖 Käyttöohjeet", expanded=False):
        st.markdown(
            """
**Mikä Kvanti on?**

Kvanti on opiskelutuutori, joka auttaa oppimaan kvantitatiivisia tutkimusmenetelmiä.

**Näin käytät:**

1. Hyväksy tietoturvahuomio.
2. Kirjoita, mitä haluat opiskella tällä oppitunnilla.
3. Kvanti aloittaa aiheen jäsentämisen.
4. Kysy, pohdi tai vastaa Kvantin kysymyksiin.
5. Avaa tarvittaessa dataset-harjoituksia.
6. Käytä toimintoa **Kertaa oppimaasi**.
7. Vie keskustelu tarvittaessa tiedostona.

Kvanti ei tee arvioitavia tehtäviä puolestasi.
            """
        )

    st.divider()

    st.markdown("### 📚 Opiskeltava asia")

    if st.session_state.study_topic:
        st.info(st.session_state.study_topic)
    else:
        st.info(
            "Opiskeltava asia kysytään aloitusnäkymässä ennen keskustelun alkua."
        )

    st.divider()

    st.markdown("### 📐 Kaavakirjasto")

    with st.expander("Näytä keskeiset kaavat", expanded=False):
        st.markdown(
            r"""
### Keskiarvo

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

Keskiarvo kuvaa havaintojen aritmeettista keskikohtaa.

---

### Otoskeskihajonta

$$
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}
$$

Keskihajonta kuvaa, kuinka paljon havainnot vaihtelevat keskiarvon ympärillä.

---

### Pearsonin korrelaatiokerroin

$$
r = \frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

$r$ vaihtelee välillä $-1$ ja $1$.

- $r > 0$: positiivinen yhteys
- $r < 0$: negatiivinen yhteys
- $r \approx 0$: ei lineaarista yhteyttä

---

### Khiin neliö -testisuure

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

missä:

- $O$ = havaittu frekvenssi
- $E$ = odotettu frekvenssi

Käytetään esimerkiksi ristiintaulukoinnissa.

---

### Riippumattomien otosten t-testin perusmuoto

$$
t = \frac{\bar{x}_1 - \bar{x}_2}
{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$

Tarkastelee kahden riippumattoman ryhmän keskiarvojen eroa.

---

### Mann-Whitneyn U-testin yksi muoto

$$
U_1 = n_1n_2 + \frac{n_1(n_1+1)}{2} - R_1
$$

missä:

- $n_1$ = ryhmän 1 koko
- $n_2$ = ryhmän 2 koko
- $R_1$ = ryhmän 1 järjestyslukujen summa
            """
        )

    st.divider()

    st.markdown("### ⚙️ Asetukset")

    with st.expander("ℹ️ Mitä asetukset tekevät?", expanded=False):
        st.markdown(
            """
**🧠 Päämalli**  
Valitsee, millä kielimallilla Kvanti vastaa.

**🌡️ Lämpötila**  
Säätää vastausten vaihtelevuutta. Suositus opiskelussa: `0.3–0.5`.

**🔁 Maks. vuoroja**  
Rajoittaa keskustelun pituutta ja kustannuksia.
            """
        )

    st.session_state["main_model"] = st.selectbox(
        "🧠 Päämalli",
        ["gpt-4o", "gpt-4o-mini"],
        index=["gpt-4o", "gpt-4o-mini"].index(st.session_state["main_model"]),
    )

    st.session_state["temperature"] = st.slider(
        "🌡️ Lämpötila",
        min_value=0.0,
        max_value=1.2,
        value=float(st.session_state["temperature"]),
        step=0.1,
    )

    st.session_state["max_turns"] = st.number_input(
        "🔁 Maks. vuoroja",
        min_value=5,
        max_value=120,
        value=int(st.session_state["max_turns"]),
        step=5,
    )

    st.divider()

    if st.button("🔄 Aloita uusi opiskeluhetki", use_container_width=True):
        if st.session_state.get("consent_given"):
            st.session_state.bot = None
            st.session_state.messages = []
            st.session_state.study_summary = None
            st.session_state.review = None
            st.session_state.dataset_exercise = None
            st.session_state.study_topic = None
            st.session_state.lesson_goal = ""
            st.session_state.lesson_goal_input = ""
            st.session_state.study_started = False
            st.rerun()
        else:
            st.error("Hyväksy ensin tietoturvahuomio.")


    st.divider()
    st.caption("📊 Kvanti v1.3 — kvantitatiivisten menetelmien tutori")


# --------------------------------------------------
# Tietoturvasuostumus
# --------------------------------------------------

st.markdown("# 📊 KVANTI")

if not st.session_state.consent_given:
    st.markdown("### 🔒 Tietoturvahuomio")

    st.markdown(
        """
Kvanti käyttää OpenAI:n kielimallia. Kaikki viestit, jotka kirjoitat,
**lähetetään OpenAI:n palveluun käsiteltäväksi**.

**Älä syötä:**

- henkilötietoja
- terveystietoja tai muita arkaluonteisia tietoja
- salasanoja tai API-avaimia
- salassa pidettävää tutkimus- tai opiskelumateriaalia

OpenAI:n tietoturva- ja yksityisyysperiaatteet:  
[https://openai.com/fi-FI/security-and-privacy/](https://openai.com/fi-FI/security-and-privacy/)

Vastuu siitä, mitä palveluun syötät, on käyttäjällä.
        """
    )

    consent = st.checkbox(
        "✅ Ymmärrän tietoturvaehdot ja sitoudun olemaan syöttämättä "
        "henkilötietoja tai arkaluonteisia tietoja.",
        value=False,
        key="consent_checkbox",
    )

    if st.button("🔓 Aloita käyttö"):
        if consent:
            st.session_state.consent_given = True
            st.rerun()
        else:
            st.error("Sinun täytyy hyväksyä tietoturvaehdot jatkaaksesi.")

    st.stop()


# --------------------------------------------------
# Aloitusnäkymä: käyttäjä syöttää opiskeltavan asian
# --------------------------------------------------

if not st.session_state.study_started:
    st.markdown(
        """
<div class="kvanti-banner">
📐 <b>Tervetuloa Kvantin opiskeluhetkeen.</b><br><br>
Kerro ensin omin sanoin:
<b>mitä tutkitte tai opiskelette tällä oppitunnilla?</b><br><br>
Voit kirjoittaa esimerkiksi menetelmän, käsitteen, tutkimusongelman
tai asian, joka tuntuu epäselvältä.
</div>
""",
        unsafe_allow_html=True,
    )

    with st.form("start_study_form", clear_on_submit=False):
        lesson_goal = st.text_area(
            "Mitä haluat opiskella tällä oppitunnilla?",
            key="lesson_goal_input",
            placeholder=(
                "Esimerkiksi: Haluan ymmärtää, milloin käytetään t-testiä "
                "ja milloin Mann-Whitneyn U-testiä.\n\n"
                "Tai: Opiskelemme p-arvoa ja hypoteesien testaamista.\n\n"
                "Tai: Tutkimme opiskelijoiden hyvinvoinnin ja opiskelutuntien yhteyttä."
            ),
            height=170,
        )

        submitted = st.form_submit_button("🚀 Aloita opiskelu")

    if submitted:
        if not lesson_goal.strip():
            st.error("Kirjoita ensin lyhyesti, mitä haluat opiskella.")
        else:
            with st.spinner("📐 Kvanti aloittaa opiskeluhetken..."):
                reset_session(lesson_goal.strip())

            st.rerun()

    st.stop()



# --------------------------------------------------
# Botin alustus varmuuden vuoksi
# --------------------------------------------------

if st.session_state.bot is None:
    if st.session_state.study_topic:
        reset_session(st.session_state.study_topic)
    else:
        st.session_state.study_started = False
        st.rerun()


bot: TutorBot = st.session_state.bot
topic_cfg = bot.config.topic_cfg
km = bot.knowledge_map


# --------------------------------------------------
# Otsikko ja banneri
# --------------------------------------------------

st.markdown(f"### `> Opiskeltava asia: {topic_cfg['display_name']}`")

st.markdown(
    f"""
<div class="kvanti-banner">
📐 <b>Kvanti</b> auttaa sinua opiskelemaan aihetta
<code>{topic_cfg['topic_text']}</code>. Kysy, pohdi ja anna Kvantin
ohjata sinua kysymyksillä ja harjoituksilla.
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Vienti Markdowniksi
# --------------------------------------------------

def build_export_md() -> str:
    current_state = bot.knowledge_map.model_dump()
    current_ratio = bot.knowledge_map.completion_ratio()

    status_label = {
        "hallussa": "Hallussa",
        "osittain": "Osittain hallussa",
        "tuntematon": "Ei vielä käsitelty",
    }

    lines = [
        f"# Kvanti — opiskelusessio",
        "",
        f"*Viety: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        f"**Opiskeltava asia:** {topic_cfg['topic_text']}  ",
        f"**Edistyminen:** {int(current_ratio * 100)} %  ",
        f"**Vuoroja:** {bot.turn_count}",
        "",
        "## Opeteltavat asiat",
        "",
    ]

    for area, status in current_state.items():
        readable_area = area.replace("_", " ").capitalize()
        readable_status = status_label.get(status, status)
        description = topic_cfg["areas"].get(area, area)

        lines.append(f"### {readable_area}")
        lines.append("")
        lines.append(f"- **Tila:** {readable_status}")
        lines.append(f"- **Kuvaus:** {description}")
        lines.append("")

    lines.append("## Keskustelu")
    lines.append("")

    for msg in st.session_state.messages:
        role = (
            "📊 **Kvanti**"
            if msg["role"] == "assistant"
            else "🧑‍🎓 **Opiskelija**"
        )
        lines.append(f"### {role}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")

    if st.session_state.dataset_exercise:
        lines.append("## Dataset-harjoitus")
        lines.append("")
        lines.append(st.session_state.dataset_exercise)
        lines.append("")

    if st.session_state.review:
        lines.append("## Kertaa oppimaasi")
        lines.append("")
        lines.append(st.session_state.review)
        lines.append("")

    if st.session_state.study_summary:
        lines.append("## Opiskeluyhteenveto")
        lines.append("")
        lines.append(st.session_state.study_summary)
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------
# Pääasettelu
# --------------------------------------------------

left_col, right_col = st.columns([2.2, 1], gap="large")


# --------------------------------------------------
# VASEN: Keskustelu ja harjoitukset
# --------------------------------------------------

with left_col:
    st.markdown("### 💬 Keskustelu Kvantin kanssa")

    st.caption(
        "🔒 Älä syötä henkilötietoja tai arkaluonteisia tietoja. "
        "Viestit lähetetään OpenAI:lle."
    )

    for msg in st.session_state.messages:
        avatar = "📊" if msg["role"] == "assistant" else "🧑‍🎓"

        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Kysy, vastaa tai pohdi ääneen..."):
        user_turns = len(
            [m for m in st.session_state.messages if m["role"] == "user"]
        )

        if user_turns >= st.session_state.max_turns:
            st.error("Maksimivuoromäärä on saavutettu. Aloita uusi opiskeluhetki.")
            st.stop()

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("📐 Kvanti miettii..."):
                reply = bot.ask(prompt)

            st.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        st.rerun()

    if st.session_state.dataset_exercise:
        st.divider()
        st.markdown("### 📊 Dataset-harjoitus")
        st.markdown(st.session_state.dataset_exercise)
        st.caption(
            "Voit vastata harjoituksen kysymyksiin keskustelukentässä. "
            "Kvanti auttaa tulkinnassa ja menetelmän valinnassa."
        )

    if st.session_state.review:
        st.divider()
        st.markdown("### 🔁 Kertaa oppimaasi")
        st.markdown(st.session_state.review)
        st.caption(
            "Voit vastata kertauskysymyksiin keskustelukentässä. "
            "Kvanti auttaa sinua syventämään ymmärrystä."
        )

    if st.session_state.study_summary:
        st.divider()
        st.markdown("### 📝 Opiskeluyhteenveto")
        st.markdown(st.session_state.study_summary)


# --------------------------------------------------
# OIKEA: Opeteltavat asiat ja toiminnot
# --------------------------------------------------

with right_col:
    st.markdown(
        """
        <div class="right-panel">
            <h3>📚 Opeteltavat asiat</h3>
            <div class="right-panel-note">
                Tämän opiskeluhetken keskeiset osa-alueet ja niiden tila keskustelun pohjalta.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_emoji = {
        "hallussa": "✅",
        "osittain": "🟡",
        "tuntematon": "⚫",
    }

    status_label = {
        "hallussa": "Hallussa",
        "osittain": "Osittain hallussa",
        "tuntematon": "Ei vielä käsitelty",
    }

    state = km.model_dump()

    with st.expander("Näytä opeteltavat asiat", expanded=False):
        for area, status in state.items():
            emoji = status_emoji.get(status, "❓")
            readable_status = status_label.get(status, status)
            description = topic_cfg["areas"].get(area, area)
            readable_area = area.replace("_", " ").capitalize()

            st.markdown(
                f"""
                <div class="learning-card">
                    <div class="learning-card-title">
                        {emoji} {readable_area}
                    </div>
                    <div class="learning-card-status">
                        <b>Tila:</b> {readable_status}
                    </div>
                    <div class="learning-card-desc">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"⏱️ Vuoro {bot.turn_count} / {bot.config.max_turns}")

    st.divider()

    st.markdown(
        """
        <div class="right-panel">
            <h3>🛠️ Toiminnot</h3>
            <div class="right-panel-note">
                Tee harjoituksia, kertaa oppimaasi, pyydä yhteenveto tai vie keskustelu.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🔁 Kertaa oppimaasi",
        use_container_width=True,
        disabled=len(st.session_state.messages) < 2,
    ):
        with st.spinner("Kvanti kokoaa kertausta..."):
            st.session_state.review = bot.review_session()

        st.rerun()

    if st.button(
        "📝 Opiskeluyhteenveto",
        use_container_width=True,
        disabled=len(st.session_state.messages) < 2,
    ):
        with st.spinner("Kvanti kokoaa yhteenvedon..."):
            st.session_state.study_summary = bot.study_summary()

        st.rerun()

    st.divider()

    st.markdown("#### 📊 Dataset-harjoitus")

    exercise_options = {
        key: value["display_name"]
        for key, value in DATASET_EXERCISES.items()
    }

    selected_exercise = st.selectbox(
        "Valitse harjoitus",
        options=list(exercise_options.keys()),
        format_func=lambda k: exercise_options[k],
    )

    if st.button(
        "📊 Avaa dataset-harjoitus",
        use_container_width=True,
    ):
        exercise_md = bot.dataset_exercise(selected_exercise)
        st.session_state.dataset_exercise = exercise_md

        bot.history.append(
            {
                "role": "assistant",
                "content": (
                    "Avasin opiskelijalle seuraavan dataset-harjoituksen:\n\n"
                    + exercise_md
                ),
            }
        )

        st.rerun()

    st.divider()

    safe_topic = "".join(
        char.lower() if char.isalnum() else "_"
        for char in (st.session_state.study_topic or "opiskelu")
    )[:40]

    filename = (
        f"kvanti_{safe_topic}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    )

    st.download_button(
        label="💾 Vie keskustelu tiedostona",
        data=build_export_md(),
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
        disabled=len(st.session_state.messages) < 2,
    )
