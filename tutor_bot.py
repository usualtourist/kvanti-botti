import json
import logging
from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Valmiit aihekuvaukset
# --------------------------------------------------
# Nämä säilytetään taustalla hyödyllisinä viitteinä, mutta käyttöliittymä
# ei enää pakota opiskelijaa valitsemaan moduulia.

TOPIC_CONFIGS = {
    "tutkimusprosessi": {
        "display_name": "Kvantitatiivisen tutkimuksen prosessi",
        "topic_text": "kvantitatiivisen tutkimuksen prosessi ja otanta",
        "opening_message": (
            "Aloitetaan kvantitatiivisen tutkimuksen prosessista. "
            "Tässä aiheessa tarkastellaan tutkimusongelman muotoilua, "
            "tutkimuksen vaiheita, aineiston kokoamista ja otantaa."
        ),
        "areas": {
            "tutkimusongelma": (
                "Tutkimustehtävän tunnistaminen ja tutkimusongelman muotoilu."
            ),
            "tutkimuksen_vaiheet": (
                "Kvantitatiivisen tutkimuksen vaiheet ideasta raportointiin."
            ),
            "aineiston_kokoaminen": (
                "Lähdemateriaalin valinta ja aineiston kokoaminen."
            ),
            "otantatekniikat": (
                "Otantatekniikat ja niiden vaikutus tulosten yleistettävyyteen."
            ),
        },
    },
    "mittaaminen": {
        "display_name": "Teoreettinen käsite ja mittaaminen",
        "topic_text": "teoreettisen käsitteen mittaaminen ja mittaustasot",
        "opening_message": (
            "Aloitetaan mittaamisesta. Tässä aiheessa tarkastellaan, "
            "miten teoreettinen käsite muutetaan mitattavaksi ja mitä "
            "mittaustasot tarkoittavat."
        ),
        "areas": {
            "kasite_ja_operationalisointi": (
                "Teoreettinen käsite ja sen operationalisointi mittariksi."
            ),
            "mittaustasot": (
                "Mittaustasot: nominaali, ordinaali, intervalli ja suhde."
            ),
            "mittarin_luotettavuus": (
                "Mittarin reliabiliteetti ja validiteetti yleisellä tasolla."
            ),
            "kytkos_tutkimusongelmaan": (
                "Mittarin valinnan kytkös tutkimusongelmaan ja tutkimustavoitteisiin."
            ),
        },
    },
    "kuvailu": {
        "display_name": "Tilastollinen kuvailu ja tunnusluvut",
        "topic_text": "tilastollisen kuvailun perusteet ja tunnusluvut",
        "opening_message": (
            "Aloitetaan tilastollisesta kuvailusta. Tässä aiheessa käsitellään "
            "frekvenssijakaumia, keskilukuja, hajontalukuja, taulukoita ja kuvaajia."
        ),
        "areas": {
            "frekvenssijakaumat": (
                "Yksi- ja useampiulotteiset frekvenssijakaumat."
            ),
            "keskiluvut": (
                "Keskiluvut, kuten keskiarvo, mediaani ja moodi sekä niiden valinta."
            ),
            "hajontaluvut": (
                "Hajontaluvut, kuten keskihajonta ja vaihteluväli."
            ),
            "taulukot_ja_kuvaajat": (
                "Tulosten esittäminen taulukoina ja kuvaajina."
            ),
        },
    },
    "paattely": {
        "display_name": "Tilastollinen päättely ja hypoteesien testaaminen",
        "topic_text": "tilastollisen päättelyn perusteet ja hypoteesien testaaminen",
        "opening_message": (
            "Aloitetaan tilastollisesta päättelystä. Tässä aiheessa käsitellään "
            "otosta, perusjoukkoa, hypoteeseja, p-arvoa ja merkitsevyystasoa."
        ),
        "areas": {
            "otos_ja_perusjoukko": (
                "Otoksen ja perusjoukon ero sekä päättelyn logiikka."
            ),
            "hypoteesit": (
                "Nolla- ja vastahypoteesin muotoilu."
            ),
            "p_arvo_ja_merkitsevyys": (
                "P-arvon tulkinta ja merkitsevyystaso."
            ),
            "virheet_ja_rajoitteet": (
                "Tilastollisten testien rajoitteet ja virhepäätelmien riskit."
            ),
        },
    },
    "riippuvuus": {
        "display_name": "Muuttujien välinen riippuvuus ja testit",
        "topic_text": (
            "muuttujien välisen riippuvuuden tutkiminen ja kahden ryhmän vertailu"
        ),
        "opening_message": (
            "Aloitetaan muuttujien välisestä riippuvuudesta ja ryhmien vertailusta. "
            "Tässä aiheessa käsitellään korrelaatiota, ristiintaulukointia, "
            "t-testiä ja Mann-Whitneyn U-testiä."
        ),
        "areas": {
            "korrelaatio": (
                "Korrelaation merkitys, tulkinta ja oletukset."
            ),
            "ristiintaulukointi": (
                "Ristiintaulukointi ja khiin neliö -testin perusperiaate."
            ),
            "ryhmien_vertailu": (
                "Kahden ryhmän eron testaaminen t-testillä ja Mann-Whitneyn U-testillä."
            ),
            "testin_valinta": (
                "Testin valinta mittaustason, jakauman ja tutkimusongelman perusteella."
            ),
        },
    },
}


# --------------------------------------------------
# Dynaaminen aihe käyttäjän syötteestä
# --------------------------------------------------

def make_custom_topic_config(custom_topic: str) -> dict:
    """
    Luo käyttäjän itse kirjoittamasta aiheesta Kvantin opiskeluaihe.

    Tätä käytetään, kun käyttöliittymässä ei ole moduulivalintaa, vaan
    opiskelija kertoo omin sanoin mitä haluaa opiskella.
    """
    topic = custom_topic.strip() or "kvantitatiivinen tutkimusmenetelmä"

    return {
        "display_name": topic,
        "topic_text": topic,
        "opening_message": (
            f"Aloitetaan aiheesta: {topic}. "
            "Kerro ensin, mikä tässä aiheessa on sinulle epäselvää "
            "tai mitä haluaisit osata oppitunnin lopuksi."
        ),
        "areas": {
            "peruskasitteet": (
                "Aiheeseen liittyvät keskeiset käsitteet ja niiden merkitys."
            ),
            "tutkimusongelma_ja_aineisto": (
                "Miten aihe liittyy tutkimusongelmaan, muuttujien valintaan ja aineistoon."
            ),
            "menetelman_valinta": (
                "Milloin ja miksi tähän aiheeseen liittyvää menetelmää tai ajattelutapaa käytetään."
            ),
            "tulkinta_ja_raportointi": (
                "Tulosten tulkinta, rajoitteet sekä raportointi taulukoiden, kuvaajien tai sanallisen tulkinnan avulla."
            ),
        },
    }


# --------------------------------------------------
# Esimerkkidatasetit
# --------------------------------------------------

DATASET_EXERCISES = {
    "korrelaatio_hyvinvointi": {
        "display_name": "Korrelaatio: opiskelutunnit ja hyvinvointi",
        "recommended_topic": "riippuvuus",
        "content": """
## Harjoitus: Korrelaatio

Tässä pienessä esimerkkiaineistossa tarkastellaan, liittyykö viikoittainen opiskeluaika koettuun hyvinvointiin.

| Havainto | Opiskelutunnit viikossa | Koettu hyvinvointi 1–10 |
|---:|---:|---:|
| 1 | 4 | 5 |
| 2 | 6 | 6 |
| 3 | 8 | 6 |
| 4 | 10 | 7 |
| 5 | 12 | 7 |
| 6 | 14 | 8 |
| 7 | 16 | 8 |
| 8 | 18 | 9 |

### Tehtävät

1. Mitkä ovat muuttujat?
2. Mikä on mahdollinen selittävä muuttuja?
3. Sopisiko tähän Pearsonin korrelaatio? Perustele.
4. Mitä positiivinen korrelaatio tarkoittaisi tässä aineistossa?
5. Mitä rajoitteita näin pienessä aineistossa on?

### Vinkki

Pearsonin korrelaatio kuvaa kahden määrällisen muuttujan lineaarista yhteyttä.
""",
    },
    "ristiintaulukointi_opintotyyvaisyys": {
        "display_name": "Ristiintaulukointi: opiskelumuoto ja tyytyväisyys",
        "recommended_topic": "riippuvuus",
        "content": """
## Harjoitus: Ristiintaulukointi

Tässä aineistossa tarkastellaan, liittyykö opiskelumuoto opiskelutyytyväisyyteen.

| Opiskelumuoto | Tyytyväinen | Ei tyytyväinen | Yhteensä |
|---|---:|---:|---:|
| Lähiopetus | 32 | 8 | 40 |
| Verkko-opetus | 24 | 16 | 40 |
| Yhteensä | 56 | 24 | 80 |

### Tehtävät

1. Mitkä ovat muuttujat?
2. Mikä on kummankin muuttujan mittaustaso?
3. Miksi ristiintaulukointi sopii tähän?
4. Miten tulkitsisit eroa prosenttien avulla?
5. Milloin $\chi^2$-testi voisi olla hyödyllinen tässä tilanteessa?

### Vinkki

Ristiintaulukointi sopii erityisesti luokitteluasteikollisten muuttujien yhteyden tarkasteluun.
""",
    },
    "ttesti_opetusryhmat": {
        "display_name": "t-testi: kahden opetusryhmän vertailu",
        "recommended_topic": "riippuvuus",
        "content": """
## Harjoitus: Riippumattomien otosten t-testi

Tarkastellaan kahden opetusryhmän tenttipisteitä.

| Ryhmä A | Ryhmä B |
|---:|---:|
| 78 | 72 |
| 82 | 75 |
| 85 | 74 |
| 80 | 70 |
| 88 | 78 |
| 84 | 73 |

### Tehtävät

1. Mikä voisi olla tutkimuskysymys?
2. Mikä on riippuva muuttuja?
3. Mikä on ryhmittelevä muuttuja?
4. Miksi riippumattomien otosten t-testi voisi sopia tähän?
5. Mitä oletuksia t-testin käyttöön liittyy?
6. Miten tulkitsisit tilanteen, jos p-arvo olisi $p < 0.05$?

### Vinkki

t-testillä voidaan tutkia, eroavatko kahden riippumattoman ryhmän keskiarvot tilastollisesti merkitsevästi toisistaan.
""",
    },
    "mannwhitney_asenne": {
        "display_name": "Mann-Whitney U: kahden ryhmän vertailu",
        "recommended_topic": "riippuvuus",
        "content": """
## Harjoitus: Mann-Whitneyn U-testi

Tarkastellaan kahden ryhmän asennepisteitä asteikolla 1–5.

| Ryhmä A | Ryhmä B |
|---:|---:|
| 3 | 2 |
| 4 | 2 |
| 4 | 3 |
| 5 | 3 |
| 4 | 2 |
| 5 | 4 |

### Tehtävät

1. Mikä voisi olla tutkimuskysymys?
2. Mikä on riippuva muuttuja?
3. Mikä on ryhmittelevä muuttuja?
4. Miksi Mann-Whitneyn U-testi voisi sopia paremmin kuin t-testi?
5. Mitä tarkoittaa, jos testi antaa tilastollisesti merkitsevän tuloksen?

### Vinkki

Mann-Whitneyn U-testiä käytetään usein, kun vertaillaan kahta riippumatonta ryhmää ja muuttuja on järjestysasteikollinen tai normaalijakaumaoletus ei ole perusteltu.
""",
    },
    "kuvailu_tunnusluvut": {
        "display_name": "Tunnusluvut: keskiluvut ja hajonta",
        "recommended_topic": "kuvailu",
        "content": """
## Harjoitus: Tunnusluvut

Tässä aineistossa on kahdeksan opiskelijan tenttipisteet.

| Opiskelija | Tenttipisteet |
|---:|---:|
| 1 | 12 |
| 2 | 15 |
| 3 | 16 |
| 4 | 16 |
| 5 | 18 |
| 6 | 19 |
| 7 | 20 |
| 8 | 28 |

### Tehtävät

1. Mikä on moodi?
2. Mikä arvo vaikuttaa poikkeavalta?
3. Kumpi olisi parempi keskiluku tässä: keskiarvo vai mediaani? Perustele.
4. Miksi hajontalukuja tarvitaan keskilukujen lisäksi?
5. Millainen kuvaaja sopisi tämän muuttujan esittämiseen?

### Vinkki

Keskiarvo voi olla herkkä poikkeaville havainnoille. Mediaani kestää usein paremmin vinoutuneita jakaumia.
""",
    },
}


# --------------------------------------------------
# KnowledgeMap
# --------------------------------------------------

class KnowledgeMap(BaseModel):
    """
    Dynaaminen osaamiskartta.

    Arvot:
    - tuntematon
    - osittain
    - hallussa
    """

    model_config = {"extra": "allow"}

    @classmethod
    def from_areas(cls, area_names: list[str]) -> "KnowledgeMap":
        return cls(**{name: "tuntematon" for name in area_names})

    def completion_ratio(self) -> float:
        values = list(self.model_dump().values())

        if not values:
            return 0.0

        mastered = sum(1 for value in values if value == "hallussa")
        return mastered / len(values)

    def is_complete(self) -> bool:
        return self.completion_ratio() >= 1.0


# --------------------------------------------------
# BotConfig
# --------------------------------------------------

@dataclass
class BotConfig:
    topic_key: str = "custom"
    custom_topic: str | None = None
    main_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_turns: int = 40

    @property
    def topic_cfg(self) -> dict:
        if self.custom_topic:
            return make_custom_topic_config(self.custom_topic)

        if self.topic_key in TOPIC_CONFIGS:
            return TOPIC_CONFIGS[self.topic_key]

        return make_custom_topic_config("kvantitatiivinen tutkimusmenetelmä")


# --------------------------------------------------
# TutorBot
# --------------------------------------------------

class TutorBot:
    def __init__(self, client: OpenAI, config: BotConfig):
        self.client = client
        self.config = config

        self.area_names = list(config.topic_cfg["areas"].keys())
        self.knowledge_map = KnowledgeMap.from_areas(self.area_names)

        self.turn_count = 0

        self.history: list[dict] = [
            {"role": "system", "content": self._system_prompt()}
        ]

    def _system_prompt(self) -> str:
        cfg = self.config.topic_cfg

        return f"""
Olet Kvanti — opiskelutuutori kvantitatiivisten menetelmien opintojaksolla.
Käyttäjä on opiskelija, joka opiskelee aihetta: "{cfg['topic_text']}".

ROOLI:
- Olet asiantunteva, selkeä ja sokraattinen tutori.
- Selität käsitteitä, annat esimerkkejä ja kysyt tarkentavia kysymyksiä.
- Et tee arvioitavia tehtäviä, kokeita tai harjoitustöitä opiskelijan puolesta.
- Autat opiskelijaa rakentamaan oman vastauksensa.

OPETUSTAPA:
1. Selitä käsitteet yksinkertaisesti ja maltillisesti.
2. Käytä ihmistieteellisiä esimerkkejä, kuten kyselyaineistot, asenteet, hyvinvointi, opiskelukokemus ja käyttäytyminen.
3. Esitä tarkentavia kysymyksiä ja pyydä opiskelijaa perustelemaan.
4. Korjaa väärinkäsitykset rauhallisesti.
5. Jos opiskelija sanoo ymmärtävänsä, kysy pieni tarkistuskysymys.
6. Käytä tarvittaessa vaiheittaista ajattelua, mutta älä tee opiskelijan tehtävää valmiiksi.

AIHEEN JÄSENTÄMINEN:
- Jos opiskelijan aihe on laaja, auta rajaamaan sitä.
- Jos aihe on epämääräinen, kysy tarkentava kysymys ennen pitkää selitystä.
- Kytke aihe kvantitatiivisen tutkimuksen näkökulmiin:
  tutkimusongelma, muuttujat, mittaaminen, aineisto, analyysimenetelmä, tulkinta ja raportointi.

VASTAUKSEN PITUUS:
- Pidä vastaukset yleensä 3–7 lauseessa.
- Jos opiskelija pyytää syvempää selitystä, voit vastata pidemmin.

MATEMATIIKKA:
- Käytä aina LaTeX-muotoa kaavoissa.
- Inline: $...$
- Block:
$$
...
$$

OHJELMISTOT:
- Voit puhua tilasto-ohjelmista yleisesti, esimerkiksi SPSS, R, Excel, Jamovi tai RStudio.
- Älä oleta tiettyä ohjelmaa, ellei opiskelija mainitse sitä.

RAJAT:
- Älä keksi lähteitä.
- Jos olet epävarma, sano se.
- Jos opiskelija pyytää suoraan valmista vastausta palautettavaan tehtävään, auta häntä kysymyksillä ja rakenteella, älä kirjoita lopullista vastausta hänen puolestaan.
- Älä pyydä henkilötietoja tai arkaluonteisia tietoja.

KIELI:
- Vastaa samalla kielellä kuin opiskelija. Oletuksena suomi.
""".strip()

    def opening_message(self) -> str:
        return self.config.topic_cfg["opening_message"]

    def ask(self, user_input: str) -> str:
        if self.turn_count >= self.config.max_turns:
            return "[Sessio on saavuttanut maksimivuoromäärän. Aloita uusi sessio.]"

        self.history.append({"role": "user", "content": user_input})
        self.turn_count += 1

        try:
            response = self.client.chat.completions.create(
                model=self.config.main_model,
                messages=self.history,
                temperature=self.config.temperature,
            )

            reply = response.choices[0].message.content

        except OpenAIError as e:
            logger.exception("OpenAI-virhe Kvantin vastauksessa")
            return f"[Virhe Kvantin vastauksessa: {e}]"

        self.history.append({"role": "assistant", "content": reply})
        self._update_knowledge_map(user_input, reply)

        return reply

    def study_summary(self) -> str:
        cfg = self.config.topic_cfg

        prompt = f"""
Olet Kvanti, opiskelutuutori.
Aihe: "{cfg['topic_text']}".

Tee opiskelijalle lyhyt opiskeluyhteenveto:
- 4–6 lausetta keskeisistä asioista, joita olette käsitelleet.
- Käytä vain tämän keskustelun sisältöä.
- Mainitse, mitkä asiat näyttävät olevan hallussa ja mitä kannattaa vielä kerrata.
- Älä keksi sisältöä, jota ei ole käsitelty.
"""

        return self._one_shot(prompt, temperature=0.3)

    def review_session(self) -> str:
        cfg = self.config.topic_cfg

        prompt = f"""
Olet Kvanti, opiskelutuutori.
Aihe: "{cfg['topic_text']}".

Auta opiskelijaa kertaamaan oppimaansa.

Tee opiskelijalle kertausta tukeva kokonaisuus:
- Aloita lyhyellä yhteenvedolla siitä, mitä keskustelussa on tähän mennessä käsitelty.
- Käytä vain tämän keskustelun sisältöä. Älä keksi uutta materiaalia.
- Nosta esiin 2–3 keskeistä käsitettä tai oivallusta.
- Lopuksi anna 3–5 lyhyttä kertauskysymystä.
- Älä anna vastauksia kertauskysymyksiin.
- Käytä samaa kieltä kuin opiskelija on käyttänyt.

Muoto:

## Kertaa oppimaasi

### Yhteenveto

...

### Keskeiset käsitteet

- ...
- ...

### Kertauskysymykset

1. ...
2. ...
"""

        return self._one_shot(prompt, temperature=0.3)

    def dataset_exercise(self, exercise_key: str) -> str:
        exercise = DATASET_EXERCISES.get(exercise_key)

        if not exercise:
            return "[Harjoitusta ei löytynyt.]"

        return exercise["content"]

    def _update_knowledge_map(self, user_input: str, assistant_reply: str) -> None:
        cfg = self.config.topic_cfg

        areas_text = "\n".join(
            f"- {name}: {description}"
            for name, description in cfg["areas"].items()
        )

        keys = ", ".join(cfg["areas"].keys())

        check_prompt = f"""
Arvioi, kuinka hyvin opiskelija on osoittanut ymmärtävänsä aiheen
"{cfg['topic_text']}" eri osa-alueet tässä keskustelussa.

Osa-alueiden määritelmät:
{areas_text}

Nykyinen tila:
{self.knowledge_map.model_dump_json(indent=2)}

Opiskelijan uusin viesti:
\"\"\"{user_input}\"\"\"

Tuutorin viimeisin vastaus kontekstina:
\"\"\"{assistant_reply}\"\"\"

Arviointisääntö:
- "hallussa": opiskelija osoittaa ymmärtävänsä asian selkeästi omin sanoin.
- "osittain": opiskelija käsittelee asiaa, mutta ymmärrys on vielä puutteellinen.
- "tuntematon": opiskelija ei ole vielä osoittanut osaamista tässä osa-alueessa.
- Älä laske osaamiseksi pelkkää kysymystä, myöntelyä tai "ymmärsin"-tyyppistä vastausta.
- Älä poista aiempaa "hallussa"-statusta, ellei opiskelija selvästi osoita väärinymmärrystä.

Palauta JSON, jossa ovat täsmälleen nämä avaimet:
{keys}

Sallitut arvot:
"tuntematon", "osittain", "hallussa"
"""

        try:
            res = self.client.chat.completions.create(
                model=self.config.judge_model,
                messages=[{"role": "system", "content": check_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            data = json.loads(res.choices[0].message.content)

            cleaned = {
                key: self._normalize_status(data.get(key, "tuntematon"))
                for key in self.area_names
            }

            self.knowledge_map = KnowledgeMap(**cleaned)

        except (OpenAIError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Knowledge map -päivitys epäonnistui: %s", e)

    def _normalize_status(self, value: str) -> str:
        allowed = {"tuntematon", "osittain", "hallussa"}

        if not isinstance(value, str):
            return "tuntematon"

        value = value.strip().lower()

        if value in allowed:
            return value

        return "tuntematon"

    def _one_shot(self, prompt: str, temperature: float = 0.3) -> str:
        msgs = self.history + [{"role": "user", "content": prompt}]

        try:
            res = self.client.chat.completions.create(
                model=self.config.main_model,
                messages=msgs,
                temperature=temperature,
            )

            return res.choices[0].message.content

        except OpenAIError as e:
            logger.exception("One-shot-kutsu epäonnistui")
            return f"[Virhe: {e}]"
