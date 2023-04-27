import logging
import requests
import spacy
import re

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from spacy.matcher import Matcher
from spacy import displacy


class Datum(BaseModel):
    key: str
    value: str
    q: str | None = None

class Metadata(BaseModel):
    file: str
    lang: str | None = None
    attrs: list[Datum] | None = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Innolab Metadata2Concept Prototype"}


@app.post("/extract/")
async def extract_metadata(data: Metadata):
    languages = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "da": "da_core_news_sm",
        "fi": "fi_core_news_sm",
        "pl": "pl_core_news_sm",
        "sv": "sv_core_news_sm",
        "nb": "nb_core_news_sm",  # Norwegian Bokmål
    }

    category_filter_patterns = [
        "^CC-BY",
        "^Self-published work",
        "images? (from|with)",
        "photographs? by",
        "supported by",
        "contributed by",
        "produced by",
        "made available",
        "donated by",
        "license",
        "upload",
    ]

    if data.lang and data.lang in languages:
        lang = data.lang
    else:
        lang = "en"

    if not data.attrs:
        attrs = query_file(data.file, lang=lang)
    else:
        attrs = data.attrs

    nlp = spacy.load(languages[lang])

    stopwords = nlp.Defaults.stop_words

    items = {}
    for attr in attrs:
        if attr.key in ["description", "title", "caption"]:
            s = clean_string(attr.value)

            for concept in extract_concepts(s, languages[lang]):
                k = concept.lower()
                if k not in items and k not in stopwords:
                    qid = query_item(concept, lang=lang)
                    items[k] = {"item": concept, "qid": qid}

        if attr.key in ["name"] and not items:
            s = clean_string(attr.value)

            for concept in extract_concepts(s, languages[lang]):
                k = concept.lower()
                if k not in items and k not in stopwords:
                    qid = query_item(concept, lang=lang)
                    items[k] = {"item": concept, "qid": qid}

        elif attr.key in ["depicts"]:
            k = attr.value.lower()
            items[k] = {"item": attr.value, "qid": attr.q}

        elif attr.key == "categories":
            cats = get_categories(attr.value)
            for cat in cats:
                if not re.match('(?:%s)' % '|'.join(category_filter_patterns), cat, re.IGNORECASE):
                    qid = query_item("Category:" + cat, lang=lang)
                    if qid:
                        items[cat] = {"item": cat, "qid": qid}

    # Filter items whether it's a disambiguation page or list page
    items = filter_administrative_pages(list(items.values()))

    # Sort by qid
    items = sorted(items, key=lambda d: d['qid'], reverse=True)

    return {
        "items": items,
    }

def filter_administrative_pages(items: list) -> list:
    filter_administrative_pages = [
        "Q4167410",  # Wikimedia disambiguation page
        "Q13406463",  # Wikimedia list pages
        "Q1071027",  #full name
        "Q101352",  # family name
        "Q110874",  # patronymic
        "Q245025",  # middle name
        "Q202444",  # given name
        "Q82799",  # name
        "Q41825",  # day of the week
        "Q47018901",  # calendar month 
        "Q427626",  # taxonomic rank 
        "Q4167836",  # Wikimedia category
    ]

    params = dict(
        action='wbgetentities',
        languages='en',
        props='claims',
        format='json',
        ids="|".join([d['qid'] for d in items if d['qid']])
    )

    response = requests.get('https://www.wikidata.org/w/api.php?', params).json()

    qnumbers = []
    if "entities" in response:
        for key, entity in response["entities"].items():
            if "id" in entity and "claims" in entity:
                for claim_key, claim_value in entity["claims"].items():
                    for claim in claim_value:
                        if "mainsnak" in claim and "datavalue" in claim["mainsnak"] and "value" in claim["mainsnak"]["datavalue"]:
                            if isinstance(claim["mainsnak"]["datavalue"]["value"], dict) and "id" in claim["mainsnak"]["datavalue"]["value"]:
                                if claim["mainsnak"]["datavalue"]["value"]["id"] in filter_administrative_pages:
                                    qnumbers.append(entity["id"])

    return [item for item in items if item["qid"] and item["qid"] not in qnumbers]

def clean_string(s: str) -> str:
    s = re.sub('<[^<]+?>', '', s)
    s = re.sub('[0-9]', '', s)
    s = re.sub('[A-Z]{3,}', ' ', s)
    s = re.sub(' {2,}', ' ', s)

    return s.strip()

def query_file(name, lang="en"):
    if name:
        attrs = []

        params = dict(
            action='wbgetentities',
            sites='commonswiki',
            format='json',
            titles="File:" + name
        )

        response = requests.get('https://commons.wikimedia.org/w/api.php?', params).json()

        p180qnums = []
        if "entities" in response:
            q = str(next(iter(response["entities"])))
            if "statements" in response["entities"][q] and "P180" in response["entities"][q]["statements"]:
                for p180 in response["entities"][q]["statements"]["P180"]:
                    if "mainsnak" in p180 and "datavalue" in p180["mainsnak"] and "value" in p180["mainsnak"]["datavalue"] and "id" in p180["mainsnak"]["datavalue"]["value"]:
                        p180qnums.append(p180["mainsnak"]["datavalue"]["value"]["id"])

        if p180qnums:
            params = dict(
                action='wbgetentities',
                props='labels',
                languages=lang,
                format='json',
                ids="|".join(p180qnums)
            )

            response = requests.get('https://www.wikidata.org/w/api.php?', params).json()

            if "entities" in response:
                for key, entity in response["entities"].items():
                    if "id" in entity and "labels" in entity and lang in entity["labels"] and "value" in entity["labels"][lang]:
                        attrs.append(Datum(key="depicts", q=entity["id"], value=entity["labels"][lang]["value"]))

        params = dict(
            action='query',
            format='json',
            prop='imageinfo',
            iiprop='extmetadata',
            iimetadataversion='latest',
            titles="File:" + name
        )

        response = requests.get('https://commons.wikimedia.org/w/api.php?', params).json()

        if "query" in response and "pages" in response["query"]:
            id = str(next(iter(response["query"]["pages"])))

            if "imageinfo" in response["query"]["pages"][id]:
                for info in response["query"]["pages"][id]["imageinfo"]:
                    if "extmetadata" in info:
                        imginfo = info["extmetadata"]

                        for k, v in imginfo.items():
                            if k == "ObjectName":
                                attrs.append(Datum(key="name", value=v["value"]))
                            elif k == "ImageDescription":
                                attrs.append(Datum(key="description", value=v["value"]))
                            elif k == "ImageTitle":
                                attrs.append(Datum(key="title", value=v["value"]))
                            elif k == "ImageCaption":
                                attrs.append(Datum(key="caption", value=v["value"]))
                            elif k == "Categories":
                                attrs.append(Datum(key="categories", value=v["value"]))

        return attrs


def query_item(name, lang="en"):
    if name:
        params = dict(
            format='json',
            action='wbgetentities',
            languages=lang,
            sites="enwiki",
            props="claims",
            titles=name.title()
        )

        response = requests.get('https://www.wikidata.org/w/api.php?', params).json()

        if "entities" in response:
            q = str(next(iter(response["entities"])))
            if q.startswith('Q'):
                if "P31" in response["entities"][q]["claims"]: # instance of
                    for snak in response["entities"][q]["claims"]["P31"]:
                        # Q15647814: Wikimedia administration category
                        if snak["mainsnak"]["datavalue"]["value"]["id"] in ["Q15647814"]:
                            return ""
                return q

    return ""

def extract_concepts(string, model):
    nlp = spacy.load(model)
    doc = nlp(string)

    matcher = Matcher(nlp.vocab)

    pattern = [
        [
            {'POS': 'PROPN', "OP": "!"},
            {'POS': 'PROPN', "DEP": "compound", "OP": "+"},
            {'POS': 'PROPN'},
            {'POS': 'PROPN', "OP": "!"},
        ]
    ]

    matcher.add('MULTIPROPN', pattern)
    matches = matcher(doc)

    compounds = []
    for match_id, start, end in matches:
        compounds.append(doc[start + 1:end - 1].text)

    subjects = [str(word) for word in doc if word.dep_ == "nsubj" or word.dep_ == "ROOT"]
    nouns = [str(word) for word in doc if word.pos_ in ['PROPN', 'NOUN'] and not any([str(word) in compound for compound in compounds])]

    candidates = list(set(x.lower() for x in nouns + subjects + compounds))

    if len(doc) <= 4:
        candidates.append(str(doc.text).lower())

    #svg = displacy.render(doc, style="dep")
    #output_path = Path("./%s.svg" % string[:10])
    #output_path.open("w", encoding="utf-8").write(svg)

    return candidates

def get_categories(l):
    if not l:
        return None

    return [s.strip() for s in l.split(";")]

