from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import json
import re
import logging
import base64
import importlib
import hashlib
import mimetypes
from io import BytesIO
from html.parser import HTMLParser
from html import escape
from urllib.parse import urljoin, urlparse
import requests
from dotenv import dotenv_values
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from datetime import datetime


def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TRANSLATION_MAX_TOKENS = 4096
RETRIEVAL_K = 3
MAX_CONTEXT_CHARS = 9000
MAX_HISTORY_MESSAGES = 6    
MAX_HISTORY_CHARS = 6000
MAX_OUTPUT_TOKENS = 1400
TARGET_PROMPT_TOKENS = 6000
RETRY_CONTEXT_CHARS = 4000
RETRY_HISTORY_CHARS = 2000
DEBUG_PROMPT_SIZE = False
BOOK_SEARCH_LIMIT = 6
BOOK_SEARCH_FETCH_LIMIT = 40

BOOK_TOPIC_TERMS = {
    "anxiety": ["anxiety", "anxious", "panic", "worry", "fear", "စိုးရိမ်ပူပန်မှု", "စိုးရိမ်စိတ်"],
    "depression": ["depression", "စိတ်ကျရောဂါ"],
    "stress": ["stress", "stress management", "စိတ်ဖိစီးမှု"],
    "trauma": ["trauma", "ptsd", "စိတ်ဒဏ်ရာ"],
    "mindfulness": ["mindfulness", "သတိပဋ္ဌာန်"],
    "self-help": ["self-help", "self help", "ကိုယ်တိုင်ကူညီ"],
    "wellbeing": ["wellbeing", "well-being", "emotional wellbeing", "emotional well-being", "စိတ်ပိုင်းဆိုင်ရာ ကောင်းမွန်မှု"],
    "sleep": ["sleep", "insomnia", "အိပ်မပျော်"],
    "emotional regulation": ["emotional regulation", "emotion regulation", "emotional control", "စိတ်ခံစားချက်ကို ထိန်းချုပ်"],
    "adolescent mental health": ["teen", "teenager", "teenagers", "adolescent", "adolescents", "ဆယ်ကျော်သက်"],
    "coping": ["coping", "coping skills", "စိတ်တည်ငြိမ်", "ရင်ဆိုင်"],
    "psychology": ["psychology", "စိတ်ပညာ"],
    "mental health": ["mental health", "psychological", "psychiatry", "စိတ်ကျန်းမာရေး"],
}
BOOK_MENTAL_TERMS = {
    "mental health", "psychology", "psychological", "psychiatry", "psychiatric",
    "anxiety", "anxious", "panic", "worry", "depression", "trauma", "ptsd",
    "mindfulness", "stress management", "self-help", "self help", "psychotherapy",
    "sleep", "insomnia", "emotional regulation", "emotion regulation", "teen", "teenager",
    "adolescent", "coping",
    "counseling", "counselling", "emotional wellbeing", "emotional well-being",
    "wellbeing", "well-being", "စိတ်ကျန်းမာရေး", "စိတ်ပညာ", "စိုးရိမ်ပူပန်မှု",
    "စိတ်ဖိစီးမှု", "စိတ်ကျရောဂါ", "စိတ်ဒဏ်ရာ",
}
BOOK_COVER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "book_covers", "burmese")
PDF_FONT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts", "NotoSansMyanmar-Regular.ttf")
DEFAULT_BOOK_COVER = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800"><rect width="600" height="800" fill="#dceae4"/><rect x="30" y="30" width="540" height="740" rx="16" fill="#b8d2c7" stroke="#24504b" stroke-width="4"/><text x="300" y="380" fill="#24504b" font-family="Arial,sans-serif" font-size="34" font-weight="700" text-anchor="middle">MENTAL HEALTH RESOURCE</text></svg>'
).decode("ascii")
BOOK_TOPIC_MAPPING = {
    "anxiety": ["Anxiety", "Stress Management", "Psychological Self-Help"],
    "stress": ["Stress Management", "Psychological First Aid", "Psychological Self-Help"],
    "depression": ["Depression", "Mood", "Psychological Self-Help"],
    "trauma": ["Trauma", "Psychological First Aid"],
    "self help": ["Psychological Self-Help", "Self-Help"],
    "psychological first aid": ["Psychological First Aid", "PFA Training"],
    "စိတ်ဖိစီးမှု": ["Stress Management", "Psychological Self-Help"],
    "စိုးရိမ်ပူပန်မှု": ["Anxiety", "Psychological Self-Help"],
    "စိတ်ကျရောဂါ": ["Depression", "Psychological Self-Help"],
    "စိတ်ပိုင်းဆိုင်ရာ": ["Psychological First Aid", "Psychological Self-Help"],
}
BOOK_REQUEST_TERMS = (
    "book", "books", "ebook", "ebooks", "e-book", "e-books", "pdf", "စာအုပ်",
)

# -------------------------------------------------------------------
# FIXED BOOK LISTS – 6 Burmese and 6 English books
# These are used when a user asks for books.
# -------------------------------------------------------------------

# Define TRUSTED_FREE_RESOURCES first

TRUSTED_FREE_RESOURCES = [
    {
        "title": "စိတ်ဖိစီးမှုကြုံတဲ့အခါ လုပ်သင့်တာတွေ လုပ်ကြမယ်",
        "author": "World Health Organization (WHO Myanmar)",
        "description": "စိတ်ဖိစီးမှုနှင့် အခက်အခဲများကို ရင်ဆိုင်နိုင်ရန် လက်တွေ့ကျသော စိတ်ကျန်းမာရေးနည်းလမ်းများ၊ grounding exercises၊ မကောင်းသောအတွေးများနှင့် ခွဲထွက်ခြင်း၊ မိမိကိုယ်ကို ကြင်နာခြင်းတို့ကို ပုံပြလမ်းညွှန်ဖြင့် သင်ကြားပေးထားသည်။",
        "language": "🇲🇲 Burmese",
        "topic": "Stress & Coping",
        "format": "PDF / Illustrated Guide / Audio",
        "resource_url": "https://www.who.int/myanmar/activities/doing-what-matters-in-times-of-stress-in-myanmar-language",
    },

    {
        "title": "စိတ်ဖိစီးမှုကို ရင်ဆိုင်ခြင်း - မြန်မာဘာသာ",
        "author": "World Health Organization (WHO Myanmar)",
        "description": "အရေးပေါ်အခြေအနေများနှင့် စိတ်ဖိစီးမှုများကြုံတွေ့ချိန်တွင် မိမိ၏စိတ်ကျန်းမာရေးကို ထိန်းသိမ်းရန် အခြေခံအကြံပြုချက်များ ပါဝင်သော မြန်မာဘာသာအရင်းအမြစ်။",
        "language": "🇲🇲 Burmese",
        "topic": "Stress",
        "format": "PDF / Information Sheet",
        "resource_url": "https://www.who.int/myanmar/emergencies/how-to-cope-with-stress",
    },

    {
        "title": "ကလေးများ၏ စိတ်ဖိစီးမှုကို ရင်ဆိုင်ခြင်း",
        "author": "World Health Organization (WHO Myanmar)",
        "description": "ကလေးများ စိတ်ဖိစီးမှုကြုံတွေ့သည့်အခါ မိဘများနှင့် စောင့်ရှောက်သူများက ကလေးများကို နားလည်ပံ့ပိုးပေးနိုင်ရန် မြန်မာဘာသာဖြင့် ပြုစုထားသော အချက်အလက်။",
        "language": "🇲🇲 Burmese",
        "topic": "Children's Mental Health",
        "format": "PDF / Information Sheet",
        "resource_url": "https://www.who.int/myanmar/emergencies/how-to-cope-with-stress",
    },

    {
        "title": "My Hero is You Too - Myanmar",
        "author": "UNICEF Myanmar",
        "description": "Aro ဆိုသည့် နဂါးငယ်လေး၏ ဇာတ်လမ်းမှတစ်ဆင့် ကလေးများအတွက် ရှုပ်ထွေးသော စိတ်ခံစားမှုများ၊ စိတ်ဖိစီးမှု၊ ဝမ်းနည်းမှုနှင့် ကျန်းမာသော coping skills များကို လွယ်ကူစွာ နားလည်စေသော mental-health flipbook။",
        "language": "🇲🇲 Burmese",
        "topic": "Children's Mental Health",
        "format": "PDF / Flipbook",
        "resource_url": "https://www.unicef.org/myanmar/reports/my-hero-you-too",
    },

    {
        "title": "When Songbirds Sing - Myanmar",
        "author": "UNICEF Myanmar",
        "description": "Nyi Nyi ၏ ဇာတ်လမ်းမှတစ်ဆင့် ဒေါသ၊ စိတ်မရှည်ခြင်း၊ mindfulness၊ ကျေးဇူးတင်ခြင်းနှင့် စိတ်ခံစားချက်များကို ကျန်းမာစွာ ကိုင်တွယ်နိုင်သည့်နည်းလမ်းများကို ကလေးများအတွက် ရှင်းပြထားသည်။",
        "language": "🇲🇲 Burmese",
        "topic": "Children's Mental Health",
        "format": "PDF / Flipbook",
        "resource_url": "https://www.unicef.org/myanmar/reports/when-songbirds-sing",
    },

    {
        "title": "Tone Tone at the Watering Hole - Myanmar",
        "author": "UNICEF Myanmar",
        "description": "ကလေးများအတွက် စိတ်ခံစားမှု၊ empathy၊ relationships နှင့် emotional wellbeing တို့ကို ဇာတ်လမ်းနှင့် ပုံများဖြင့် လေ့လာနိုင်စေရန် ပြုစုထားသော mental-health flipbook။",
        "language": "🇲🇲 Burmese",
        "topic": "Children's Mental Health",
        "format": "PDF / Flipbook",
        "resource_url": "https://www.unicef.org/myanmar/reports/tone-tone-watering-hole",
    },

    {
        "title": "My Hero is You - Myanmar",
        "author": "Inter-Agency Standing Committee (IASC) / UNICEF Myanmar",
        "description": "COVID-19 နှင့် အခြားခက်ခဲသောအခြေအနေများအတွင်း ကလေးများနှင့် မိသားစုများက ကြောက်ရွံ့မှု၊ စိုးရိမ်မှုနှင့် စိတ်ခံစားချက်များကို နားလည်ရင်ဆိုင်နိုင်ရန် ဖန်တီးထားသော မြန်မာဘာသာ storybook။",
        "language": "🇲🇲 Burmese",
        "topic": "Children & Family Mental Health",
        "format": "PDF / Storybook",
        "resource_url": "https://www.unicef.org/myanmar/reports/my-hero-you",
    },

    {
        "title": "Myanmar Mental Health and Self-Care Materials",
        "author": "SUNI-SEA",
        "description": "မြန်မာဘာသာဖြင့် စိတ်ကျန်းမာရေး၊ self-care၊ psychological wellbeing နှင့် psychological first aid ဆိုင်ရာ ပညာပေးပစ္စည်းများကို စုစည်းပေးထားသော အခမဲ့အရင်းအမြစ်စုစည်းမှု။",
        "language": "🇲🇲 Burmese",
        "topic": "Mental Health & Self-Care",
        "format": "PDF / Educational Materials",
        "resource_url": "https://www.suni-sea.org/en/resources/iec-materials-on-mental-health-and-self-care-myanmar/",
    },

    {
        "title": "Mental Health Resources - Myanmar",
        "author": "MHPSS Working Group Myanmar",
        "description": "မြန်မာနိုင်ငံအတွင်း စိတ်ကျန်းမာရေးနှင့် psychosocial support အတွက် အသုံးပြုနိုင်သော toolkits၊ activity materials နှင့် community-support resources များကို စုစည်းပေးထားသည်။",
        "language": "🇲🇲 Burmese / English",
        "topic": "Psychosocial Support",
        "format": "Resource Library / PDF",
        "resource_url": "https://www.mhpssmyanmar.org/resources",
    },

    {
        "title": "Addressing Mental Health in Myanmar",
        "author": "World Health Organization",
        "description": "မြန်မာနိုင်ငံ၏ စိတ်ကျန်းမာရေးအခြေအနေ၊ စိတ်ကျန်းမာရေးဝန်ဆောင်မှုများနှင့် စိတ်ကျန်းမာရေးစောင့်ရှောက်မှု တိုးတက်ရေးဆိုင်ရာ WHO country report။",
        "language": "🇬🇧 English",
        "topic": "Mental Health in Myanmar",
        "format": "PDF / Country Report",
        "resource_url": "https://www.who.int/publications/i/item/9789290210207",
    },

    {
        "title": "Mental Health Care for Immigrants - Burmese",
        "author": "WayAhead / NSW Multicultural Health Communication Service",
        "description": "မြန်မာဘာသာဖြင့် စိတ်ကျန်းမာရေးအကူအညီလိုအပ်နေမှုကို သိရှိနိုင်ရန်၊ professional support ရှာဖွေရန်နှင့် စိတ်ကျန်းမာရေးဝန်ဆောင်မှုများကို နားလည်နိုင်ရန် ပြုစုထားသော အခမဲ့လမ်းညွှန်။",
        "language": "🇲🇲 Burmese",
        "topic": "Mental Health Awareness",
        "format": "PDF / Guide",
        "resource_url": "https://www.mhcs.health.nsw.gov.au/publications/mental-health-care-for-immigrants/burmese",
    },
]



FIXED_BURMESE_BOOKS = [
{
    "id": "burmese-free-mental-health-1",
    "title": "Health Messenger Magazine No. 28 - Special Issue on Mental Health",
    "authors": ["Aide Medicale Internationale (AMI)"],
    "description": "စိတ်ကျန်းမာရေး၊ စိတ်ဖိစီးမှုကို ကိုင်တွယ်ခြင်း၊ counselling၊ psychosocial support၊ စိတ်ကျန်းမာရေးဆိုင်ရာ အသိပညာပေးအကြောင်းအရာများ ပါဝင်သော မြန်မာ/အင်္ဂလိပ် နှစ်ဘာသာ PDF စာစောင်။",
    "language": "🇲🇲 Burmese / English",
    "category": "Mental Health Education",
    "format": "PDF / Magazine",
    "availability": "FREE",
    "cover_url": "assets/book_covers/burmese-free-mental-health-1.jpg",
    "read_url": "https://www.burmalibrary.org/en/health-messenger-magazine-no-28-special-issue-on-mental-health",
    "resource_url": "https://www.burmalibrary.org/en/health-messenger-magazine-no-28-special-issue-on-mental-health",
    "source": "Aide Medicale Internationale / Online Burma-Myanmar Library",
    "score": 18,
    "is_free": True,
    "published_year": 2005
},

    {
        "id": "burmese-1",
        "title": "စိတ်ဝင်္ကပါမှ လွတ်မြောက်ခြင်း",
        "authors": ["မွန်ဟော်စီ (Mon Halsey)"],
        "description": "A guide to recognizing and escaping from mental traps and cognitive mazes, focusing on mental healing and self-actualization.",
        "language": "🇲🇲 Burmese",
        "category": "Psychology / Self-Help",
        "format": "Book",
        "availability": "PAID",
        "cover_url": "assets/book_covers/escaping_mind_maze.jpg",
        "read_url": "https://nbinet3.ncl.edu.tw/record=b20081611~S10*chi",
        "resource_url": "https://nbinet3.ncl.edu.tw/record=b20081611~S10*chi",
        "source": "National Central Library (Taiwan)",
        "score": 25,
        "is_free": False,
        "published_year": 2025
    },

    {
        "id": "burmese2",
        "title": "လက်စွဲစာအုပ် ၃ - စိတ်ကျန်းမာရေးနှင့် ကောင်းမွန်သောဘဝကို မြှင့်တင်ခြင်း",
        "authors": ["EarthRights"],
        "description": "စိတ်ကျန်းမာရေး၊ စိတ်ကျန်းမာရေးပြဿနာနှင့် wellbeing တို့ကို မြန်မာဘာသာဖြင့် ရှင်းပြထားပြီး စိတ်ပိုင်းဆိုင်ရာကျန်းမာရေးနှင့် ကောင်းမွန်သောဘဝကို မြှင့်တင်ရန် လက်တွေ့အသုံးချနိုင်သော အကြောင်းအရာများ ပါဝင်သည်။",
        "language": "🇲🇲 Burmese",
        "category": "Mental Health & Wellbeing",
        "format": "PDF / Handbook",
        "availability": "FREE",
        "cover_url": "assets/book_covers/burmese2.jpg",
        "read_url": "https://earthrights.org/wp-content/uploads/2024/12/Book-5-Wellbeing-and-Psychosocial-Interventions-for-Earth-Rights-Defenders-in-and-From-Myanmar.pdf",
        "resource_url": "https://earthrights.org/wp-content/uploads/2024/12/Book-5-Wellbeing-and-Psychosocial-Interventions-for-Earth-Rights-Defenders-in-and-From-Myanmar.pdf",
        "source": "EarthRights",
        "score": 20,
        "is_free": True,
        "published_year": 2023,
    },

    {
        "id": "burmese3",
        "title": "စိတ်ကျန်းမာရေးဆေးပညာအဘိဓာန်နှင့် ဝေါဟာရရှင်းတမ်း",
        "authors": ["ဒေါက်တာအုန်းကျော်"],
        "description": "စိတ်ကျန်းမာရေးနှင့်သက်ဆိုင်သော ဝေါဟာရ ၂,၂၀၀ ကျော်၊ စိတ်ရောဂါအမည်များ၊ ရောဂါလက္ခဏာများ၊ ကုထုံးများနှင့် ဆေးဘက်ဆိုင်ရာဝေါဟာရများကို အင်္ဂလိပ်-မြန်မာဖြင့် ရှင်းလင်းဖော်ပြထားသည်။",
        "language": "🇲🇲 Burmese / English",
        "category": "Psychiatry & Mental Health",
        "format": "Book / Dictionary",
        "availability": "PAID",
        "cover_url": "assets/book_covers/burmese3.jpg",
        "read_url": "https://todaybooks.com.mm/shop/book/1342",
        "resource_url": "https://todaybooks.com.mm/shop/book/1342",
        "source": "TODAY Book Store",
        "score": 19,
        "is_free": False,
        "published_year": 2025,
    },

    {
    "id": "burmese4",
    "title": "မိမိစိတ်ကို ထိန်းချုပ်ခြင်းဆိုင်ရာအနုပညာ",
    "authors": ["မြမြဆွေ (စိတ်ပညာ)"],
    "description": "မိမိစိတ်ကို နားလည်ထိန်းချုပ်ခြင်း၊ ပူဆွေးသောကကို စီမံခြင်း၊ mindfulness၊ စိတ်ကျခြင်းနှင့် စိတ်ကျန်းမာရေးကောင်းမွန်စေရေးတို့ကို လေ့လာနိုင်သော မြန်မာဘာသာစိတ်ပညာစာအုပ်။",
    "language": "🇲🇲 Burmese",
    "category": "Emotional Wellbeing & Psychology",
    "format": "Book",
    "availability": "PAID",
    "cover_url": "assets/book_covers/burmese4.jpg",
    "read_url": "https://todaybooks.com.mm/shop/book/970",
    "resource_url": "https://todaybooks.com.mm/shop/book/970",
    "source": "TODAY Book Store",
    "score": 20,
    "is_free": False,
    "published_year": 2025,
},

    {
    "id": "burmese5",
    "title": "ချစ်သူနဲ့ စိတ်ကျန်းမာရေး",
    "authors": ["စူးရှမေ"],
    "description": "မိမိနှင့် မိမိချစ်သူ၏ စိတ်ဒဏ်ရာများကို နားလည်ခြင်း၊ relationship များအတွင်း စိတ်ကျန်းမာရေးကို ဂရုစိုက်ခြင်းနှင့် ကျန်းမာသောဆက်ဆံရေး တည်ဆောက်ခြင်းအကြောင်း ရေးသားထားသော မြန်မာဘာသာစာအုပ်။",
    "language": "🇲🇲 Burmese",
    "category": "Relationships & Mental Health",
    "format": "Book",
    "availability": "PAID",
    "cover_url": "assets/book_covers/burmese5.jpg",
    "read_url": "https://todaybooks.com.mm/shop/book/1230",
    "resource_url": "https://todaybooks.com.mm/shop/book/1230",
    "source": "TODAY Book Store",
    "score": 20,
    "is_free": False,
    "published_year": 2025,
},

    {
        "id": "burmese6",
        "title": "Burmese - Mental Health Care for Immigrants",
        "authors": ["WayAhead"],
        "description": "မြန်မာဘာသာဖြင့် စိတ်ကျန်းမာရေးအကူအညီ လိုအပ်နေမှုကို သတိပြုနိုင်ရန်၊ ကုသမှုရွေးချယ်စရာများနှင့် အကူအညီရှာဖွေနိုင်သည့် နည်းလမ်းများကို ရှင်းပြထားသော အခမဲ့ PDF။",
        "language": "🇲🇲 Burmese",
        "category": "Mental Health Awareness",
        "format": "PDF / Guide",
        "availability": "FREE",
        "cover_url": "assets/book_covers/burmese6.jpg",
        "read_url": "https://www.mhcs.health.nsw.gov.au/publications/mental-health-care-for-immigrants/burmese",
        "resource_url": "https://www.mhcs.health.nsw.gov.au/publications/mental-health-care-for-immigrants/burmese",
        "source": "WayAhead / NSW Multicultural Health Communication Service",
        "score": 18,
        "is_free": True,
        "published_year": 2021,
    },

    {
        "id": "burmese7",
        "title": "Health Messenger Magazine No. 28 - Mental Health Special Issue",
        "authors": ["Aide Medicale Internationale"],
        "description": "စိတ်ကျန်းမာရေး၊ စိတ်ဖိစီးမှုကို ကိုင်တွယ်ခြင်း၊ counselling၊ စိတ်ပိုင်းဆိုင်ရာနှင့် လူမှုရေးဆိုင်ရာ အထောက်အပံ့များအကြောင်း ပါဝင်သော မြန်မာ/အင်္ဂလိပ် နှစ်ဘာသာထုတ် စာစောင်။",
        "language": "🇲🇲 Burmese / English",
        "category": "Mental Health Education",
        "format": "PDF / Magazine",
        "availability": "FREE",
        "cover_url": "assets/book_covers/burmese7.jpg",
        "read_url": "https://www.burmalibrary.org/en/health-messenger-magazine-no-28-special-issue-on-mental-health",
        "resource_url": "https://www.burmalibrary.org/en/health-messenger-magazine-no-28-special-issue-on-mental-health",
        "source": "Online Burma/Myanmar Library",
        "score": 17,
        "is_free": True,
        "published_year": 2005,
    },
]

FIXED_ENGLISH_BOOKS = [
    {
        "id": "eng1",
        "title": "A Beginner's Guide to Being Mental",
        "authors": ["Natasha Devon"],
        "description": "A comprehensive guide to mental health from one of the UK's foremost experts, debunking and demystifying the full spectrum of mental health.",
        "language": "English",
        "category": "Mental Health Education",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "A Beginner's Guide to Being Mental.png",
        "read_url": "https://www.panmacmillan.com/authors/natasha-devon/a-beginners-guide-to-being-mental/9781509882229",
        "resource_url": "https://www.panmacmillan.com/authors/natasha-devon/a-beginners-guide-to-being-mental/9781509882229",
        "source": "Pan Macmillan",
        "score": 20,
        "is_free": False,
        "published_year": 2018,
    },
    {
        "id": "eng2",
        "title": "My Lovely Wife",
        "authors": ["Mark Lukach"],
        "description": "A powerful memoir of one man's overwhelming love for his wife through mental illness and psychosis.",
        "language": "English",
        "category": "Memoir",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "My Lovely Wife.png",
        "read_url": "https://www.panmacmillan.com/authors/mark-lukach/my-lovely-wife/9781509805969",
        "resource_url": "https://www.panmacmillan.com/authors/mark-lukach/my-lovely-wife/9781509805969",
        "source": "Pan Macmillan",
        "score": 19,
        "is_free": False,
        "published_year": 2018,
    },
    {
        "id": "eng3",
        "title": "Iron Hope",
        "authors": ["James Lawrence"],
        "description": "Develops mental toughness to accomplish anything; essential reading for anybody with goals.",
        "language": "English",
        "category": "Motivation",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "Iron Hope by James Lawrence.png",
        "read_url": "https://www.worldofbooks.com/en-au/products/iron-hope-book-james-lawrence-9781035062249?sku=GOR014315170",
        "resource_url": "https://www.worldofbooks.com/en-au/products/iron-hope-book-james-lawrence-9781035062249?sku=GOR014315170",
        "source": "Pan Macmillan",
        "score": 18,
        "is_free": False,
        "published_year": 2025,
    },
    {
        "id": "eng4",
        "title": "Ten Times Calmer",
        "authors": ["Dr Kirren Schnack"],
        "description": "An Oxford-trained NHS clinical psychologist offers a practical first-aid kit of tools to help you understand anxiety, manage stress, tackle trauma, and find calm – with short exercises and clinically proven tips.",
        "language": "English",
        "category": "Anxiety / Stress Management",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "Ten Times Calmer by Kirren Schnack.png",
        "read_url": "https://www.worldofbooks.com/en-au/products/ten-times-calmer-book-kirren-schnack-9781035013623?sku=GOR014714578",
        "resource_url": "https://www.worldofbooks.com/en-au/products/ten-times-calmer-book-kirren-schnack-9781035013623?sku=GOR014714578",
        "source": "Pan Macmillan",
        "score": 18,
        "is_free": False,
        "published_year": 2023,
    },
    {
        "id": "eng5",
        "title": "Furiously Happy",
        "authors": ["Jenny Lawson"],
        "description": "A hilarious, outrageous memoir about the author's lifelong battle with mental illness – exploring crippling depression and anxiety while embracing joy in fantastic, joy-filled ways.",
        "language": "English",
        "category": "Memoir / Mental Health",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "Gemini_Generated_Image_uyovjeuyovjeuyov.jpeg",
        "read_url": "https://www.panmacmillan.com/authors/jenny-lawson/furiously-happy/9781529080704",
        "resource_url": "https://www.panmacmillan.com/authors/jenny-lawson/furiously-happy/9781529080704",
        "source": "Pan Macmillan",
        "score": 18,
        "is_free": False,
        "published_year": 2021,
    },
    {
        "id": "eng6",
        "title": "Maybe I Don't Belong Here",
        "authors": ["David Harewood"],
        "description": "A powerful and provocative memoir from the critically acclaimed actor – charting his life from working-class Birmingham to Hollywood, his psychotic breakdown, and the impact of everyday racism on Black mental health.",
        "language": "English",
        "category": "Memoir / Race & Mental Health",
        "format": "Book",
        "availability": "REFERENCE",
        "cover_url": "Maybe I Don't Belong Here.png",
        "read_url": "https://www.panmacmillan.com/authors/david-harewood/maybe-i-dont-belong-here/9781529064155",
        "resource_url": "https://www.panmacmillan.com/authors/david-harewood/maybe-i-dont-belong-here/9781529064155",
        "source": "Pan Macmillan",
        "score": 18,
        "is_free": False,
        "published_year": 2021,
    },
]
# -------------------------------------------------------------------
# Original resource databases – using the fixed lists for now
# -------------------------------------------------------------------
BURMESE_BOOKS = FIXED_BURMESE_BOOKS
MENTAL_HEALTH_BOOKS = FIXED_ENGLISH_BOOKS
VERIFIED_FREE_MENTAL_HEALTH_RESOURCES = TRUSTED_FREE_RESOURCES


def _complete_book_record(book):
    record = dict(book)
    record.setdefault("availability", "REFERENCE")
    record.setdefault("link_label", "View Official Resource")
    record.setdefault("format", record.get("type", "Book"))
    record.setdefault("cover_url", "")
    return record


BURMESE_BOOKS = [_complete_book_record(book) for book in BURMESE_BOOKS]
MENTAL_HEALTH_BOOKS = [_complete_book_record(book) for book in MENTAL_HEALTH_BOOKS]
VERIFIED_FREE_MENTAL_HEALTH_RESOURCES = [
    {
        "title": book.get("title", ""),
        "authors": [book.get("source", "Official resource")],
        "description": book.get("description", ""),
        "language": book.get("language", "English"),
        "category": book.get("category", "Mental Health"),
        "source": book.get("source", "Verified open-access source"),
        "resource_url": book.get("resource_url", ""),
        "cover_url": book.get("cover_url", ""),
        "access_type": "FREE",
        "readable": True,
        "source_type": "verified_open_access",
    }
    for book in BURMESE_BOOKS + TRUSTED_FREE_RESOURCES
]

working_dir = os.path.dirname(os.path.realpath(__file__))

# Define LOCAL_IMAGE_FOLDER at the module level
LOCAL_IMAGE_FOLDER = os.path.join(working_dir, "images")


def _find_pdf_font(filename):
    search_roots = [
        os.path.join(working_dir, "fonts"),
        working_dir,
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for directory, _, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(directory, filename)
    return ""


def _register_pdf_fonts():
    regular_path = _find_pdf_font("NotoSansMyanmar-Regular.ttf") or PDF_FONT_PATH
    bold_path = _find_pdf_font("NotoSansMyanmar-Bold.ttf")
    if not os.path.isfile(regular_path):
        raise FileNotFoundError(
            "Myanmar PDF font not found. Expected fonts/NotoSansMyanmar-Regular.ttf"
        )
    if "NotoMyanmar" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("NotoMyanmar", regular_path))
    if bold_path and "NotoMyanmarBold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("NotoMyanmarBold", bold_path))
    elif "NotoMyanmarBold" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("NotoMyanmarBold", regular_path))
    return regular_path


def _is_pdf_emoji(character):
    codepoint = ord(character)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2300 <= codepoint <= 0x23FF
        or 0xFE0F == codepoint
        or 0x200D == codepoint
    )


def _pdf_text(text, font_name="NotoMyanmar"):
    value = str(text or "")
    glyphs = pdfmetrics.getFont(font_name).face.charWidths
    value = "".join(
        character
        for character in value
        if not _is_pdf_emoji(character) or ord(character) in glyphs
    )
    return escape(value, quote=False).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")


def build_chat_history_pdf(messages):
    regular_path = _register_pdf_fonts()
    logger.info("Building chat-history PDF with Myanmar font: %s", regular_path)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Genesis Care - Conversation History",
        author="Genesis Care",
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "MyanmarHeading",
        parent=styles["Heading1"],
        fontName="NotoMyanmarBold",
        fontSize=16,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "MyanmarBody",
        parent=styles["BodyText"],
        fontName="NotoMyanmar",
        fontSize=10,
        leading=16,
        wordWrap="CJK",
        spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "EnglishBody",
        parent=body_style,
        fontName="NotoMyanmarBold",
        fontSize=11,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )
    story = [
        Paragraph(_pdf_text("Genesis Care - Conversation History"), heading_style),
        Paragraph(_pdf_text(datetime.now().strftime("Date: %Y-%m-%d %H:%M:%S")), body_style),
        Spacer(1, 8),
    ]
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).capitalize()
        content = str(message.get("content", ""))
        story.append(Paragraph(_pdf_text(role), label_style))
        story.append(Paragraph(_pdf_text(content), body_style))
        story.append(Spacer(1, 6))
    document.build(story)
    return buffer.getvalue()


env_path = os.path.join(working_dir, ".env")
env_config = {}
if os.path.exists(env_path):
    env_config = {key: value for key, value in dotenv_values(env_path).items() if value}
    if not env_config or not {"GROQ_API_KEY", "YOUTUBE_API_KEY", "GOOGLE_BOOKS_API_KEY"}.intersection(env_config):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                parsed_env = json.load(f)
            if isinstance(parsed_env, dict):
                env_config = {key: str(value).strip() for key, value in parsed_env.items() if value}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            env_config = {}
config_path = os.path.join(working_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
else:
    config_data = {}
GROQ_API_KEY = str(
    os.environ.get("GROQ_API_KEY") or env_config.get("GROQ_API_KEY") or config_data.get("GROQ_API_KEY", "")
).strip()
YOUTUBE_API_KEY = str(
    os.environ.get("YOUTUBE_API_KEY") or env_config.get("YOUTUBE_API_KEY") or config_data.get("YOUTUBE_API_KEY", "")
).strip()
GOOGLE_BOOKS_API_KEY = str(
    os.environ.get("GOOGLE_BOOKS_API_KEY") or env_config.get("GOOGLE_BOOKS_API_KEY") or config_data.get("GOOGLE_BOOKS_API_KEY", "")
).strip()
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
else:
    os.environ.pop("GROQ_API_KEY", None)

# FastAPI app and Pydantic model for API input
app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    chat_history: list[dict] = Field(default_factory=list)

# FastAPI route for chatbot
@app.post("/chat")
async def chatbot(request: MessageRequest):
    message = request.message

    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Groq API key is missing. Add it to config.json or set GROQ_API_KEY before running the app.")

    if is_translation_request(message):
        response = translate_previous_response(
            message,
            request.chat_history,
            ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=TRANSLATION_MAX_TOKENS),
        )
    # Check for sensitive topics
    elif contains_sensitive_topics(message):
        response = "It seems you may be experiencing distress. Please know that help is available. If you feel unsafe or overwhelmed, consider calling emergency services immediately. For ongoing support, please contact a verified local mental-health service. We strongly encourage you to seek professional help."
    else:
        # Setup vectorstore (same as Streamlit code)
        vectorstore = setup_vectorstore()

        # Setup the conversational chain (same as Streamlit code)
        conversational_chain = chat_chain(vectorstore)

        # Get response from the conversational chain
        response = conversational_chain({"question": message, "chat_history": request.chat_history})["answer"]

    return {"response": _sanitize_assistant_response(response)}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """Your role is to provide safe, compassionate, evidence-based mental health information while maintaining professional boundaries.
Your role is to provide safe, compassionate, evidence-based mental health information while maintaining professional boundaries.

IMPORTANT MEDIA RULE:
The chatbot has a separate media recommendation system that searches YouTube and displays videos/songs in video cards.

Media requests are ALLOWED when the user asks for:
- Motivational videos
- Motivational songs
- Healing songs
- Relaxing songs
- Calming music
- Stress-relief music
- Anxiety-relief or calming videos
- Meditation videos
- Mindfulness videos
- Breathing exercise videos
- Sleep or relaxation music
- Funny videos that are appropriate for improving mood
- Funny Shorts that are appropriate and safe
- K-pop songs when requested for mood, relaxation, enjoyment, or emotional wellbeing
- Romantic songs when requested for emotional wellbeing, positive mood, relaxation, or enjoyment
- Songs by a specific artist when the request is related to mood, relaxation, motivation, emotional wellbeing, or supportive entertainment
- English or Burmese songs/videos that are appropriate for the user's requested purpose
- Religious or spiritual supportive content when requested
- Other safe entertainment media when it can reasonably support positive mood or emotional wellbeing

Therefore, DO NOT refuse a media request simply because the requested song, singer, music genre, funny video, or entertainment content is not itself a mental-health educational topic.

For example:

User: "Give me some motivational songs."
Correct behavior:
- Accept the request.
- Give a short supportive response.
- Let the media system search for appropriate motivational songs.
- Do not say that you can only discuss mental health.

User: "Give me some healing songs."
Correct behavior:
- Accept the request.
- Treat it as supportive emotional-wellbeing media.
- Let the media system search for suitable songs.

User: "Give me some Taylor Swift songs to help me relax."
Correct behavior:
- Accept the request.
- Let the media system search for Taylor Swift songs.
- Do not refuse simply because Taylor Swift is an entertainment artist.

User: "Show me some Burmese motivational songs."
Correct behavior:
- Accept the request.
- The media system should search specifically for Burmese-language motivational songs.

User: "Show me some English calming songs."
Correct behavior:
- Accept the request.
- The media system should search specifically for English-language calming songs.

User: "Give me some funny videos."
Correct behavior:
- Accept the request as mood-supportive entertainment.
- Only allow safe and appropriate funny content.
- Never intentionally recommend sexual, nude, pornographic, explicit, or otherwise inappropriate content.

LANGUAGE RULE FOR MEDIA:
The media system must respect the language requested by the user.

If the user asks in English or explicitly requests English:
- Search for English-language videos or songs.
- Do not intentionally return Burmese-language media.

If the user asks in Burmese or explicitly requests Burmese/Myanmar:
- Search for Burmese-language videos or songs.
- Do not intentionally return English-language media unless the user asks for English.

If the user asks for a specific artist:
- Search for that artist's requested content.
- Do not require the artist to be in a fixed predefined list.
- Users may request any singer or artist naturally.

Examples:
- "Give me Taylor Swift songs"
- "Give me Ed Sheeran songs"
- "Give me IU songs"
- "Give me BLACKPINK songs"
- "Give me some Burmese songs by Ni Ni Khin Zaw"
- "Show me songs by Sai Sai Kham Leng"

The media system should use the user's requested artist, singer, genre, language, and purpose when constructing the YouTube search query.

IMPORTANT COUNTRY / CONTENT FILTER:
Do NOT intentionally recommend Indian videos or songs when the user is requesting English or Burmese content.

Do not intentionally search for:
- Indian-language videos
- Indian entertainment content
- Bollywood content
- Hindi songs
- Indian regional-language songs

unless the user explicitly asks for them.

IMPORTANT SAFETY FILTER FOR MEDIA:
Never intentionally recommend or search for:
- Nudity
- Naked content
- Pornographic content
- Sexual content
- Explicit sexual videos
- Sexually suggestive videos
- Fetish content
- Adult entertainment
- Graphic violence
- Dangerous challenges
- Self-harm content
- Suicide-related content
- Content encouraging harmful behavior

Funny videos and Funny Shorts must be family-friendly, safe, and non-sexual.

YouTube media should be treated as supportive entertainment, not medical treatment.

GENERAL RESPONSE RULES:
1. First determine what the user is asking.
2. If the user is asking a normal mental-health question, answer using the provided mental-health context and vector database.
3. If the user is asking for supportive media, songs, videos, or entertainment, DO NOT refuse merely because the media itself is not educational.
4. Allow safe media recommendations when they are related to mood, relaxation, motivation, emotional wellbeing, coping, stress relief, or supportive entertainment.
5. Keep normal assistant responses concise and easy to read.
6. Use bullet points when useful.
7. Use only a few emojis when appropriate.
8. Do not claim that YouTube was searched unless actual YouTube API results were successfully obtained.
9. Do not invent video titles, URLs, channels, durations, or video IDs.
10. The media system, not the LLM response, is responsible for obtaining real YouTube video metadata.

MENTAL HEALTH CONTEXT RULE:
For mental-health questions, prioritize the provided context and vector database.

If the user asks a mental-health question and relevant context exists:
- Provide a helpful response based on that context.
- Be empathetic and supportive.
- Avoid diagnosis.
- Avoid prescribing medication.
- Avoid claiming to be a therapist or medical professional.

If a normal question has no connection to mental health and is not a supported media request:
- Politely explain that this chatbot focuses on mental health and supportive wellbeing media.
MYANMAR MENTAL HEALTH RESOURCE RULE:

This chatbot is primarily designed for users in Myanmar.

When providing mental-health support resources:

- Prioritize verified Myanmar-based mental-health and psychosocial-support services.
- Do NOT recommend India's Tele-MANAS service.
- Do NOT mention the Indian Tele-MANAS numbers 14416 or 1800-891-4416 for users in Myanmar.
- Do NOT assume that an India-based resource is appropriate for a Myanmar user.
- Do NOT invent Myanmar helpline numbers, Telegram usernames, Facebook pages, websites, addresses, operating hours, or service availability.

Myanmar resources that may be considered when appropriate include:
- Jue Jue's Safe Space
- Aung Mental Health Initiative

Only provide specific contact information when it is available from a verified source or from the application's trusted resource database.

If the user explicitly says they are in another country, use resources appropriate to that country instead.

If the user's location is unknown and crisis resources are needed, avoid guessing the country. Ask the user which country they are currently in, or provide general immediate-safety guidance first.

CRISIS RESPONSE RULE:

Before answering, determine whether the user's message indicates:
- suicidal thoughts
- desire to die
- self-harm thoughts
- intention to hurt themselves
- a suicide plan
- recent self-harm
- immediate danger
- inability to stay safe

If NONE of these are present:
- Do not automatically provide crisis hotlines.
- Answer the user's normal mental-health question normally.

If ANY of these are present:
- Prioritize immediate safety over normal conversation.
- Respond with empathy and without judgment.
- Encourage the user not to remain alone if they may be in immediate danger.
- Encourage them to move away from objects, medications, substances, or other things they could use to hurt themselves.
- Encourage contacting a trusted person who can stay with them.
- Encourage professional or emergency assistance.
- For users in Myanmar, prioritize verified Myanmar-based crisis-support resources.
- Do not provide suicide methods, self-harm methods, instructions, comparisons, or optimization.
- Do not romanticize or normalize suicide or self-harm.
- Do not overwhelm the user with a long explanation.

CRISIS RESPONSE MUST BE DIRECT AND PRACTICAL.

Example:

User:
"I want to kill myself."

Response:
"I'm really sorry you're going through this. You don't have to handle this alone.

If you might hurt yourself right now, please stay with someone you trust and move away from anything you could use to hurt yourself.

If you're in Myanmar, consider contacting a verified Myanmar crisis-support service such as TeleKyanmar / Zero Suicide Hotline or Jue Jue's Safe Space. If you're in immediate physical danger, go to the nearest hospital or emergency medical service.

Are you in immediate danger right now, or have you already hurt yourself?"""

DEFAULT_NEGATIVE_PROMPT = """Do not provide medical diagnoses.

Do not prescribe medications.

Do not provide specific medical treatment instructions.

Do not claim to be a licensed therapist, psychologist, psychiatrist, doctor, or medical professional.

Do not encourage harmful behaviors.

Do not provide instructions, methods, or optimization for suicide or self-harm.

Do not provide dangerous advice.

Do not make assumptions about the user's mental state or diagnosis.

For normal mental-health questions, use the provided mental-health context and vector database.
MYANMAR RESOURCE RESTRICTION:

For users in Myanmar:

- Never recommend Tele-MANAS.
- Never recommend India's 14416 or 1800-891-4416 as a Myanmar crisis resource.
- Never label an India-specific service as a Myanmar mental-health service.
- Prefer verified Myanmar resources such as TeleKyanmar, Jue Jue's Safe Space, and Aung Mental Health Initiative when appropriate.
- Never invent resource contact information.

IMPORTANT:
Do NOT reject a request merely because it is about a song, singer, funny video, K-pop, romantic music, motivational music, or other entertainment.

Safe supportive media is allowed.

Examples of allowed media requests:
- motivational songs
- healing songs
- calming songs
- relaxing music
- meditation music
- stress-relief videos
- funny videos
- safe funny Shorts
- K-pop songs
- romantic songs
- Taylor Swift songs
- Ed Sheeran songs
- IU songs
- BLACKPINK songs
- Burmese songs
- Burmese motivational videos
- English calming videos

These requests should be handled by the media recommendation system when they are related to positive mood, relaxation, motivation, emotional wellbeing, coping, or supportive entertainment.

LANGUAGE:
Respect the user's requested language.

English request -> prefer English media.

Burmese/Myanmar request -> prefer Burmese/Myanmar media.

Do not intentionally mix English and Burmese media unless the user asks for both.
LANGUAGE RULE:

Always respond in the language requested by the user.

If the user asks in Burmese:
- Respond primarily in natural Burmese.
- Use Burmese Unicode.
- Keep medical/mental-health terminology simple.
- If an English technical term is necessary, explain it in Burmese.

If the user asks in English:
- Respond in English.

Do not unnecessarily mix Burmese and English.

When providing Myanmar mental-health resources to a Burmese-speaking user, explain the resource in Burmese.

COUNTRY FILTER:
Do not intentionally recommend Indian videos or songs unless the user explicitly requests Indian content.

Avoid intentionally returning:
- Bollywood
- Hindi songs
- Indian regional-language entertainment
- Indian entertainment channels

unless explicitly requested.
    
SAFETY FILTER:
Never intentionally recommend:
- nudity
- naked content
- pornography
- sexual content
- explicit sexual content
- sexually suggestive entertainment
- adult entertainment
- graphic violence
- dangerous challenges
- self-harm content
- suicide-promoting content
- content that encourages harmful behavior

Funny videos must be safe and family-friendly.

MEDIA DATA:
Never invent YouTube video IDs.

Never invent YouTube URLs.

Never invent titles.

Never invent channels.

Never invent durations.

Never claim that YouTube was searched unless actual YouTube API results were returned.

The YouTube API/media-search component is responsible for retrieving real video metadata.

The assistant should not create fake media cards or fake video information."""
COMPACT_SYSTEM_PROMPT = """You are a compassionate mental-health support assistant. Use the supplied context when relevant and answer clearly. Do not diagnose, prescribe medication, claim to be a clinician, invent facts, or give dangerous advice. If the context is not relevant, say you can only discuss mental-health topics.

Keep ordinary answers concise, normally under 450 words. Use headings and bullet or numbered lists when useful. Do not use Markdown tables, HTML tags, browser-generated anchor links, localhost URLs, or empty Markdown links. Before returning, complete the final sentence and ensure the response ends naturally. If space is limited, shorten the answer instead of stopping halfway.

LANGUAGE AND BURMESE RESPONSE RULES:
Match the user's requested language. Answer Burmese questions primarily in natural Burmese Unicode. Use clear, simple Myanmar wording and do not translate English word-for-word. Do not unnecessarily mix English and Burmese; briefly explain an English mental-health term in Burmese only when it is necessary.

For ordinary Burmese mental-health questions, be warm, calm, respectful, and non-judgmental. First acknowledge the user's experience, then provide 2-5 realistic practical steps such as rest, regular meals and water, gentle movement, breathing, grounding (လက်ရှိပတ်ဝန်းကျင်ကို အာရုံစိုက်ခြင်း), mindfulness (လက်ရှိအချိန်မှာ ဖြစ်နေတဲ့အရာတွေကို သတိထားခြင်း), journaling, taking a break, speaking with a trusted person, or seeking professional support. Ask at most one gentle follow-up question when it would genuinely help.

Do not diagnose or prescribe medication. Use wording such as "ဒီလိုခံစားနေရတာက စိတ်ဖိစီးမှုနဲ့ ဆက်စပ်နိုင်ပါတယ်။" Do not claim that advice will cure the user. For normal Burmese advice, use a short heading, a brief supportive introduction, numbered sections or bullets, and a short supportive closing. Do not automatically add emergency information or a long disclaimer to ordinary questions.

CRISIS OVERRIDE:
If the user expresses suicidal thoughts, self-harm intent, immediate danger, or intent to harm another person, prioritize crisis support over normal advice. Respond in the user's language, including Burmese when appropriate. Use short, direct safety-focused language; encourage contacting a trusted person and appropriate emergency or verified crisis support. Never provide methods or invented Myanmar contact details, and never recommend India-specific crisis services as Myanmar services.

BURMESE RESPONSE COMPLETENESS RULE:
When responding in Burmese, complete every sentence, word, recommendation, bullet point, numbered item, table row, parenthesis, and Markdown element. Never stop halfway or leave unfinished text. If the answer may become too long, shorten it before writing rather than truncating it. Prefer numbered sections or bullets instead of tables for long Burmese answers. Silently verify that the response ends naturally before returning it.
MYANMAR RESOURCE RULE:
For users in Myanmar, use verified Myanmar mental-health resources.
Never recommend India's Tele-MANAS or 14416/1800-891-4416 as Myanmar resources.
Never invent crisis contact information.
Prefer verified Myanmar resources such as TeleKyanmar, Jue Jue's Safe Space, and Aung Mental Health Initiative when appropriate.

MARKDOWN FORMATTING RULE:
Never use HTML tags such as <br>, <p>, <div>, <ul>, <li>, or <table> in the response. Never write literal HTML inside Markdown tables. Use Burmese Unicode characters directly. Keep Markdown tables simple and valid, and use "•" separators for multiple items inside a table cell."""

TEXT_RESPONSE_SYSTEM_PROMPT = COMPACT_SYSTEM_PROMPT + """

TEXT RESPONSE POLICY:
Always answer the user's actual request with a meaningful text response. A separate media system handles videos, songs, and media cards; never invent or describe media metadata, cards, thumbnails, URLs, IDs, channels, or durations. Media must never replace mental-health guidance when the user also asks for advice.

CONVERSATION MEMORY:
Use the available conversation history when the user says that, this, it, the previous answer, your previous response, translate that, explain that, summarize that, continue, or make it shorter. If the referenced assistant response exists, use it directly and never ask the user to provide it again or claim it is unavailable.

LANGUAGE SWITCHING:
Recognize translate to Burmese, translate into Burmese, in Burmese, switch to Burmese, speak Burmese, answer in Burmese, မြန်မာလို, မြန်မာလို ပြောပါ, မြန်မာလို ဖြေပါ, and မြန်မာဘာသာနဲ့ ပြောပါ as Burmese language commands. Translate or restate the immediately preceding assistant response when the command refers to it, then continue using Burmese until another language is requested. Use natural Burmese Unicode, simple terminology, and minimal English mixing.

COMPLETENESS:
Return complete sentences, list items, headings, Markdown, and parentheses. If space is limited, shorten the answer before returning it. For crisis indicators, prioritize short, direct safety guidance and verified resources over normal advice or media.

AUTHORITATIVE RESPONSE POLICY:
Always answer the user's actual request with meaningful text. Do not diagnose, prescribe medication, claim to be a clinician, or claim that advice will cure a condition. Media or book systems must never replace the main mental-health response, and do not invent media or book metadata.

CONVERSATION MEMORY:
Use the immediately preceding relevant assistant response when the user refers to "that", "this", "it", the previous answer, your previous response, or asks to translate, explain, summarize, shorten, or continue it. If that response exists, do not ask the user to provide it again or claim it is unavailable.

TRANSLATION AND LANGUAGE SWITCHING:
Translate the immediately preceding assistant response, preserving meaning and useful structure, for explicit commands such as "Translate to Burmese", "Translate that to Burmese", "Translate this to Burmese", "မြန်မာလို ဘာသာပြန်ပါ", "မြန်မာလို ပြန်ပေးပါ", or "Burmese please". Do not translate the user's command or generate an unrelated answer.
Treat "Give me mental health advice in Burmese", "မြန်မာလို စိတ်ကျန်းမာရေးအကြံပေးပါ", and "မြန်မာလို ဖြေပါ" as new Burmese requests, not translation requests. Use natural Burmese Unicode and simple terminology. Treat "Switch to Burmese" as a language-switch instruction and continue in Burmese; use the previous response as source text only when the user clearly asks to translate or refers to it.

MYANMAR RESOURCE POLICY:
Use only verified resource information available to the application. Do not invent contact details, websites, addresses, hours, or availability. Do not recommend TeleKyanmar. Do not recommend India's Tele-MANAS or the numbers 14416 or 1800-891-4416 as Myanmar resources. Do not describe WHO Myanmar, UNFPA Myanmar, Myanmar Red Cross Society, or MHPSS Working Group Myanmar as crisis hotlines unless verified data explicitly confirms that classification. Use Jue Jue's Safe Space and Aung Mental Health Initiative only according to verified application data.

CRISIS PRIORITY:
If the user expresses suicidal thoughts, self-harm intent, immediate danger, inability to stay safe, or serious intent to harm another person, safety guidance takes priority over normal advice, media, books, or translation. Use short, direct, empathetic guidance and only verified resources. Never provide methods, instructions, optimization, or encouragement for harm.

FINAL CHECK:
Before returning, verify that the response answers the actual request, uses conversation history when needed, distinguishes translation from a new Burmese request, follows the requested language, avoids invented information, remains useful without media or books, and ends naturally.
"""
def contains_sensitive_topics(question):
    sensitive_keywords = [
        'suicide', 'self-harm', 'self harm', 'kill myself', 'kill', 'end my life',
        'want to die', 'hurting myself', 'cutting', 'overdose', 'harm myself',
        'suicidal', 'self injury', 'self-injury', 'self mutilation',
        'self-mutilation', 'suicidal thoughts', 'suicidal ideation',
        'harmful behaviors', 'hurting myself', 'ending it all', 'no reason to live',
        'can\'t go on', 'want to disappear', 'don\'t want to exist',
        'ကိုယ့်ကိုယ်ကို သတ်', 'ကိုယ့်ကိုယ်ကို ထိခိုက်', 'မနေချင်တော့',
        'သေချင်', 'ကိုယ့်ကိုယ်ကို နာကျင်အောင်'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in sensitive_keywords)


def _translation_target(question):
    lowered = question.lower()
    if re.search(
        r"\b(burmese|myanmar)\b|\b(?:in|into|to|speak|answer)\s+burmese\b|"
        r"မြန်မာ|ဗမာ|မြန်မာလို",
        question,
        re.IGNORECASE,
    ):
        return "Burmese"
    if re.search(r"\b(english|อังกฤษ)\b|အင်္ဂလိပ်|အင်္ဂလိပ်လို", question, re.IGNORECASE):
        return "English"
    if re.search(r"\b(chinese|japanese|korean|spanish|french|german|thai)\b", lowered):
        match = re.search(r"\b(chinese|japanese|korean|spanish|french|german|thai)\b", lowered)
        return match.group(1).capitalize() if match else None
    return None


def is_translation_request(question):
    direct_burmese_request = bool(re.search(
        r"\b(?:give|provide|explain|tell|write|share)\b.*\b(?:in|using)\s+burmese\b|"
        r"\b(?:mental\s+health|advice|information|guidance|question|books?|help)\b.*\b(?:in|using)\s+burmese\b|"
        r"မြန်မာလို\s+စိတ်ကျန်းမာရေး",
        question,
        re.IGNORECASE,
    ))
    if direct_burmese_request or re.search(r"မြန်မာလို\s+ဖြေပါ", question):
        return False
    direct_burmese_request = bool(re.search(
        r"\b(?:give|provide|explain|tell|write|share)\b.*\b(?:in|using)\s+burmese\b|"
        r"\b(?:mental\s+health|advice|information|guidance|question|books?|help)\b.*\b(?:in|using)\s+burmese\b",
        question,
        re.IGNORECASE,
    ))
    if direct_burmese_request:
        return False
    has_translation_intent = bool(re.search(
        r"\b(translate|translation|switch|change|convert)\b|"
        r"\bburmese\s+please\b|"
        r"\b(?:in|into|to|speak|answer)\s+burmese\b|"
        r"မြန်မာလို|မြန်မာဘာသာနဲ့|ဘာသာပြန်|ဘာသာစကား|လိုပြောင်း",
        question,
        re.IGNORECASE,
    ))
    return has_translation_intent


def _conversation_language_instruction(chat_history):
    for entry in reversed(chat_history or []):
        if not isinstance(entry, dict) or entry.get("role") != "user":
            continue
        content = str(entry.get("content", "")).strip()
        target = _translation_target(content)
        if target:
            return f"Continue responding in {target} unless the user explicitly requests another language."
        if content and _is_burmese_request(content):
            return "Respond primarily in natural Burmese Unicode."
        return ""
    return ""


def _normalize_llm_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def _llm_finish_reason(response):
    metadata = getattr(response, "response_metadata", {}) or {}
    if isinstance(metadata, dict):
        reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        if reason:
            return str(reason).lower()
    return ""


def _ensure_response_full_stop(response):
    text = str(response or "").strip()
    if text and text[-1] not in ".?!\u104a\u104b\u104f\u3002\uFF01\uFF1F\u2026":
        return f"{text}."
    return text


def _sanitize_assistant_response(response):
    """Remove browser-only artifacts before displaying or returning assistant text."""
    text = _normalize_llm_content(response)
    text = re.sub(r"\[\s*\]\(https?://(?:localhost|127\.0\.0\.1)(?::\d+)?/#[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _ensure_response_full_stop(text)


def translate_previous_response(question, chat_history, llm):
    target_language = _translation_target(question)
    previous_response = next(
        (
            str(entry.get("content", "")).strip()
            for entry in reversed(chat_history or [])
            if isinstance(entry, dict) and entry.get("role") == "assistant" and str(entry.get("content", "")).strip()
        ),
        None,
    )
    if not target_language:
        return "Which language would you like the previous response translated into?"
    if not previous_response:
        return "There is no previous assistant response to translate."

    prompt = f"""Translate the ENTIRE immediately preceding assistant response below into {target_language}.
TRANSLATION AND PREVIOUS-RESPONSE RULE:
Treat the preceding assistant response as the source text. Preserve its meaning, useful structure, headings, bullets, and all relevant details. Do not translate the user's request instead. Do not generate a new unrelated answer, summarize, or ask the user to provide the previous response again. If the preceding response is present, never claim that it is unavailable.
Translate EVERY English sentence, clause, heading, label, bullet point, numbered point, question, example, and closing sentence. Do not leave any ordinary English sentence untranslated. Keep the same order, meaning, number of points, and paragraph or list structure. Preserve proper nouns, URLs, names, and unavoidable technical terms when they do not have a natural translation.
When the translation is Burmese, never use HTML tags such as <br>, <p>, <div>, <ul>, <li>, or <table>. Use Burmese Unicode characters directly. Keep Markdown tables simple and valid, and use "•" separators for multiple items inside a table cell. Do not place literal HTML inside Markdown tables.
Use as many output tokens as necessary to finish the translation. Return only a complete, faithful, direct translation. Do not summarize, shorten, omit, explain, add advice, retrieve information, or include the original text. Do not stop early.

Assistant response:
{previous_response}"""
    translated_parts = []
    response = llm.invoke(prompt)
    translated_parts.append(_normalize_llm_content(getattr(response, "content", response)))

    for _ in range(2):
        if _llm_finish_reason(response) not in {"length", "max_tokens", "max_output_tokens"}:
            break
        continuation_prompt = f"""Continue the {target_language} translation exactly where it stopped.
    Return only the missing translated text. Complete the current sentence, word, list item, table row, parenthesis, and Markdown element before stopping. Do not repeat any text already provided and do not add commentary.

Translation already returned:
{''.join(translated_parts)}"""
        response = llm.invoke(continuation_prompt)
        translated_parts.append(_normalize_llm_content(getattr(response, "content", response)))

    return _sanitize_assistant_response("".join(translated_parts))

from html import escape
import re
import streamlit as st

def render_video_cards_from_list(
    videos, category="Recommended Videos", language="en"
):
    """Render YouTube results as clean thumbnail cards with full-width action buttons."""
    if not videos:
        return

    display_videos = [
        v
        for v in videos
        if v.get("embeddable") is True
        and v.get("embed_url")
        and re.fullmatch(r"[A-Za-z0-9_-]{6,}", str(v.get("video_id", "")).strip())
    ][:6]

    if not display_videos:
        return

    cards_html = []
    for video in display_videos:
        title = video.get("title", "Recommended Video")
        video_id = str(video.get("video_id", "")).strip()
        safe_title = escape(title, quote=True)
        
        # High quality thumbnail image (shows original video thumbnail/watermarks without YouTube UI overlays)
        thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        cards_html.append(
            f"""
        <div class="video-card-item">
            <div class="video-thumb-wrap watch-btn" data-video-id="{video_id}">
                <img class="video-thumb" src="{escape(thumb_url, quote=True)}" alt="{safe_title}" loading="lazy" />
            </div>
            <div class="video-action-wrap">
                <button class="watch-now-button watch-btn" type="button" data-video-id="{video_id}">
                    <span class="watch-icon">▶</span> Watch Now
                </button>
            </div>
        </div>
        """
        )

    html = f"""
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background: transparent;
        }}
        
        .video-container {{
            width: 100%;
            display: flex;
            justify-content: center;
            padding: 0 4px;
        }}
        
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
        }}
        
        @media (max-width: 850px) {{
            .video-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
                max-width: 700px;
            }}
        }}
        
        @media (max-width: 550px) {{
            .video-grid {{
                grid-template-columns: 1fr;
                max-width: 400px;
            }}
        }}
        
        .video-card-item {{
            display: flex;
            flex-direction: column;
            gap: 8px; /* Distinct visual separation between video and button */
            width: 100%;
        }}
        
        .video-thumb-wrap {{
            position: relative;
            width: 100%;
            height: 180px;
            overflow: hidden;
            background: #000000;
            border-radius: 10px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .video-thumb-wrap:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
        }}
        
        .video-thumb {{
            display: block;
            width: 100%;
            height: 100%;
            object-fit: cover;
            border: 0;
        }}
        
        .video-action-wrap {{
            width: 100%;
            display: flex;
        }}
        
        .watch-now-button {{
            width: 100%;
            box-sizing: border-box;
            padding: 8px 12px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            letter-spacing: 0.2px;
            line-height: 1.3;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
        }}
        
        .watch-now-button:hover {{
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }}
        
        .watch-now-button:active {{
            transform: translateY(0);
        }}
        
        .watch-icon {{
            font-size: 0.65rem;
        }}
        
        /* Modal Styles */
        .yt-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 10000;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background: rgba(15, 23, 42, 0.92);
            backdrop-filter: blur(12px);
        }}
        
        .yt-modal-content {{
            position: relative;
            width: min(960px, 95vw);
            max-width: 95vw;
            aspect-ratio: 16 / 9;
            background: #000000;
            border-radius: 12px;
            overflow: visible;
            box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
        }}
        
        .yt-frame {{
            position: relative;
            z-index: 1;
            width: 100%;
            height: 100%;
            border: 0;
            border-radius: 12px;
        }}
        
        .yt-close {{
            position: absolute;
            top: 14px;
            right: 0;
            bottom: auto;
            z-index: 2;
            width: 44px;
            height: 44px;
            border: none;
            border-radius: 50%;
            background: rgba(0, 0, 0, 0.7);
            color: #ffffff;
            font-size: 24px;
            cursor: pointer;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(4px);
        }}
        
        .yt-close:hover {{
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.1) rotate(90deg);
        }}
        
        .yt-modal-content::before {{
            content: "Loading...";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: rgba(255, 255, 255, 0.4);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
            letter-spacing: 0.5px;
            z-index: 0;
        }}
        
        @media (max-width: 768px) {{
            .yt-modal {{
                padding: 12px;
            }}
            
            .yt-modal-content {{
                width: 98vw;
                border-radius: 8px;
            }}
            
            .yt-close {{
                top: 10px;
                right: 10px;
                bottom: auto;
                width: 36px;
                height: 36px;
                font-size: 18px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .yt-modal {{
                padding: 8px;
            }}
            
            .yt-modal-content {{
                width: 100vw;
                border-radius: 4px;
            }}
            
            .yt-close {{
                top: 8px;
                right: 8px;
                bottom: auto;
                width: 32px;
                height: 32px;
                font-size: 16px;
            }}
        }}
    </style>
    
    <div class="video-container">
        <div class="video-grid">
            {''.join(cards_html)}
        </div>
    </div>
    
    <div class="yt-modal" id="yt-modal" role="dialog" aria-modal="true" aria-label="Video player">
        <div class="yt-modal-content">
            <button class="yt-close" id="yt-close" type="button" aria-label="Close video">×</button>
            <iframe class="yt-frame" id="yt-frame" title="Selected video" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
        </div>
    </div>
    
    <script>
        const modal = document.getElementById('yt-modal');
        const frame = document.getElementById('yt-frame');
        const closeBtn = document.getElementById('yt-close');
        
        const closeModal = () => {{
            modal.style.display = 'none';
            frame.src = '';
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
        }};
        
        const openModal = (videoId) => {{
            if (!videoId) return;
            frame.src = 'https://www.youtube.com/embed/' + videoId + '?autoplay=1&rel=0&modestbranding=1&iv_load_policy=3&enablejsapi=1&controls=1&showinfo=0';
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
            document.body.style.paddingRight = '0px';
        }};
        
        document.querySelectorAll('.watch-btn').forEach((btn) => {{
            btn.addEventListener('click', function(event) {{
                const videoId = this.getAttribute('data-video-id');
                openModal(videoId);
                event.stopPropagation();
            }});
        }});
        
        closeBtn.addEventListener('click', closeModal);
        
        modal.addEventListener('click', function(event) {{
            if (event.target === event.currentTarget) {{
                closeModal();
            }}
        }});
        
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape' && modal.style.display === 'flex') {{
                closeModal();
            }}
        }});
    </script>
    """
    st.components.v1.html(html, height=520, scrolling=False)


MEDIA_CATEGORIES = [
    "Motivation", "Anxiety", "Stress Management", "Relaxation", "Breathing Exercises",
    "Meditation", "Mindfulness", "Sleep", "Self-Compassion", "Resilience",
    "Emotional Wellbeing", "Personal Growth", "Sadness Support", "Anger Management",
    "Loneliness Support", "Self-Esteem", "Confidence", "Focus & Concentration",
    "Student Wellbeing", "Workplace Wellbeing", "Burnout", "Positive Thinking",
    "Mental Health Education", "Religious & Spiritual Support", "Sermons / Religious Talks",
    "Buddhist Dhamma", "Christian Sermons", "Islamic Lectures", "Hindu Spiritual Talks",
    "Jewish Sermons", "Sikh Spiritual Talks", "Songs", "Music Videos", "Motivational Songs",
    "Healing Songs", "Calming Songs", "Relaxing Songs", "Peaceful Songs", "Sad Songs",
    "Funny Videos", "Funny Shorts", "Motivational Videos", "K-pop", "Artist Music",
    "Romantic Songs", "Positive Songs", "Nature Relaxation",
]
RELIGIONS = ["Buddhism", "Christianity", "Islam", "Hinduism", "Judaism", "Sikhism"]
RELIGION_TERMS = {
    "Buddhism": ["buddhist", "buddhism", "ဗုဒ္ဓ", "တရား"],
    "Christianity": ["christian", "christianity"], "Islam": ["islam", "islamic", "muslim"],
    "Hinduism": ["hindu", "hinduism"], "Judaism": ["jewish", "judaism"],
    "Sikhism": ["sikh", "sikhism", "katha"],
}
RELIGION_CATEGORIES = {
    "Buddhism": "Buddhist Dhamma", "Christianity": "Christian Sermons", "Islam": "Islamic Lectures",
    "Hinduism": "Hindu Spiritual Talks", "Judaism": "Jewish Sermons", "Sikhism": "Sikh Spiritual Talks",
}
MENTAL_QUERIES = {
    "Motivation": "mental health motivation resilience self compassion", "Anxiety": "mental health anxiety coping techniques",
    "Stress Management": "stress management mental health coping", "Relaxation": "relaxation stress anxiety calming exercise",
    "Breathing Exercises": "guided breathing exercise anxiety stress relaxation", "Meditation": "guided mindfulness meditation mental health",
    "Self-Compassion": "self compassion mental health self criticism",
}
RELIGIOUS_QUERIES = {
    "Christianity": "Christian sermon hope encouragement peace", "Islam": "Islamic lecture patience hope peace",
    "Hinduism": "Hindu spiritual discourse peace wellbeing", "Judaism": "Jewish sermon resilience hope wellbeing",
    "Sikhism": "Sikh katha spiritual peace resilience",
}


EXPLICIT_MEDIA_TERMS = {
    "porn", "pornography", "nude", "nudity", "naked", "sex tape", "sexual video",
    "explicit sex", "xxx", "nsfw", "onlyfans", "erotic", "hardcore", "blowjob",
}
INDIAN_MEDIA_TERMS = {
    "india", "indian", "hindi", "bollywood", "tamil", "telugu", "malayalam",
    "bengali", "kannada", "marathi", "punjabi", "desi",
}


def _is_burmese_request(question):
    return bool(re.search(r"\b(burmese|myanmar)\b|[\u1000-\u109f]", question, re.IGNORECASE))


def contains_burmese(text):
    return bool(re.search(r"[\u1000-\u109F]", str(text or "")))


def _detect_media_language(question, artist=None, detected_language=None):
    if _is_burmese_request(question):
        return "my"
    return "my" if detected_language == "my" else "en"


def _extract_artist_from_request(question):
    patterns = [
        r"(?:songs?|music|videos?)\s+by\s+(.+?)(?:\s+(?:songs?|music|videos?|tracks?|shorts?)\b|$)",
        r"(?:give me|show me|play|find|get|listen to)\s+(.+?)\s+(?:songs?|music|videos?|tracks?|shorts?)\b",
        r"(?:give me|show me|play|find|get|listen to)\s+(.+?)\s+သီချင်း",
    ]
    ignored = {"some", "something", "a", "an", "the", "funny", "motivational", "healing", "relaxing", "calming", "romantic", "love", "positive", "sweet", "cute", "popular", "k-pop", "kpop"}
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            continue
        words = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?\"").split()
        while words and words[0].lower() in ignored:
            words.pop(0)
        while words and words[-1].lower() in ignored:
            words.pop()
        if words:
            return " ".join(words)
    burmese_artist = re.search(r"^\s*([^၊။!?]+?)\s+သီချင်း", question)
    return burmese_artist.group(1).strip() if burmese_artist else None


def _is_explicit_media_result(video):
    snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
    text = " ".join(str(video.get(field, "")) for field in ("title", "description", "channel"))
    text += " " + " ".join(str(snippet.get(field, "")) for field in ("title", "description", "channelTitle"))
    text = text.lower()
    return any(term in text for term in EXPLICIT_MEDIA_TERMS)


def _is_indian_content(video):
    snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
    text = " ".join(str(video.get(field, "")) for field in ("title", "description", "channel"))
    text += " " + " ".join(str(snippet.get(field, "")) for field in ("title", "description", "channelTitle"))
    text = text.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in INDIAN_MEDIA_TERMS)


def is_safe_media_result(video, media_request):
    if _is_explicit_media_result(video):
        return False
    if not media_request.get("explicit_indian_content") and _is_indian_content(video):
        return False
    snippet = video.get("snippet", {}) if isinstance(video.get("snippet"), dict) else {}
    text = " ".join(str(video.get(field, "")) for field in ("title", "description", "channel"))
    text += " " + " ".join(str(snippet.get(field, "")) for field in ("title", "description", "channelTitle"))
    if media_request.get("language") == "my":
        if not re.search(r"[\u1000-\u109f]", text) and not media_request.get("artist"):
            return False
    elif not media_request.get("explicit_indian_content") and re.search(r"[\u0900-\u097f\u0b80-\u0bff\u0c00-\u0c7f\u0d00-\u0d7f]", text):
        return False
    return True


def _fallback_media_request(question):
    lowered = question.lower()
    religion = next((name for name, terms in RELIGION_TERMS.items()
                     if any(term in lowered or term in question for term in terms)), None)
    language = _detect_media_language(question)
    artist = None if religion else _extract_artist_from_request(question)
    category = RELIGION_CATEGORIES.get(religion, "")
    category_terms = {"Motivation": "motivat", "Anxiety": "anxiety", "Stress Management": "stress", "Relaxation": "calm",
                      "Breathing Exercises": "breath", "Meditation": "meditat", "Mindfulness": "mindful",
                      "Self-Compassion": "self-compassion", "Resilience": "resilien", "Sleep": "sleep", "Burnout": "burnout"}
    if not religion:
        if re.search(r"\bfunny\b", lowered) and re.search(r"\bshorts?\b", lowered):
            category = "Funny Shorts"
        elif re.search(r"\bfunny\b", lowered):
            category = "Funny Videos"
        elif re.search(r"\b(k-pop|kpop)\b", lowered):
            category = "K-pop"
        elif re.search(r"\b(motivational?|motivation)\b.*\b(song|music)\b", lowered):
            category = "Motivational Songs"
        elif re.search(r"\bhealing\b.*\b(song|music)\b", lowered):
            category = "Healing Songs"
        elif re.search(r"\bcalming\b.*\b(song|music)\b", lowered):
            category = "Calming Songs"
        elif re.search(r"\b(relaxing|relaxation|peaceful)\b.*\b(song|music)\b", lowered):
            category = "Relaxing Songs"
        elif re.search(r"\b(romantic|love)\b.*\b(song|music)\b", lowered):
            category = "Romantic Songs"
        elif re.search(r"\b(sad)\b.*\b(song|music)\b", lowered):
            category = "Sad Songs"
        elif re.search(r"\b(positive|uplifting)\b.*\b(song|music)\b", lowered):
            category = "Positive Songs"
        else:
            category = next((name for name, term in category_terms.items() if term in lowered), "")
        if artist:
            category = "Artist Music"
        elif not category and re.search(r"သီချင်း", lowered):
            category = "Songs"
        elif not category and re.search(r"\b(video|videos|watch|show|listen|sermon|lecture|talks?)\b|ဗီဒီယို", lowered, re.IGNORECASE):
            category = "Mental Health Education"
    query_defaults = {"Funny Videos": "funny videos", "Funny Shorts": "funny shorts", "Motivational Songs": "motivational songs uplifting music", "Healing Songs": "healing songs relaxing music", "Calming Songs": "calming songs peaceful music", "Relaxing Songs": "relaxing songs calming music", "Romantic Songs": "romantic love songs", "Sad Songs": "sad songs emotional music", "Positive Songs": "positive uplifting songs", "K-pop": "K-pop songs", "Music Videos": "music videos"}
    query = "ဗုဒ္ဓဘာသာ တရားတော် မေတ္တာ စိတ်ငြိမ်းချမ်းမှု" if religion == "Buddhism" else None
    if not query and artist:
        query = f"{artist} {('songs' if language == 'en' else 'သီချင်း')}"
    if not query:
        query = query_defaults.get(category, RELIGIOUS_QUERIES.get(religion, MENTAL_QUERIES.get(category, "mental health wellbeing educational support")))
    if language == "my" and not re.search(r"[\u1000-\u109f]", query):
        query = f"{question} {query}"
    explicit_indian = bool(re.search(r"\b(india|indian|hindi|tamil|telugu|malayalam|bengali|kannada|marathi|punjabi|bollywood)\b", lowered))
    if language == "en" and not explicit_indian:
        query += " -India -Hindi -Tamil -Telugu -Bollywood"
    return {"is_requested": bool(category), "category": category, "religion": religion, "artist": artist, "youtube_search_query": query, "max_results": 6, "language": language, "explicit_indian_content": explicit_indian, "source": question}


def detect_media_request(question, llm):
    prompt = f"Return only valid JSON keys is_requested, category, religion, artist, youtube_search_query, max_results, language, explicit_indian_content. Allowed categories: {', '.join(MEDIA_CATEGORIES)}. Allowed religions: {', '.join(RELIGIONS)} or null. Artist may be any singer, band, group, or Burmese artist named by the user, otherwise null. Detect explicit requests for songs, music, artists, videos, Shorts, funny content, or mental-health media. Use language en or my based on the request; Burmese script or Burmese/Myanmar media uses my. Set explicit_indian_content true only when the user explicitly asks for Indian content. Never create URLs, video IDs, or metadata. For Buddhism use Burmese search terms. Request: {question}"
    try:
        raw = str(llm.invoke(prompt).content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}
        if result.get("is_requested") and result.get("category") in MEDIA_CATEGORIES:
            result["religion"] = result.get("religion") if result.get("religion") in RELIGIONS else None
            result["artist"] = str(result.get("artist")).strip() if result.get("artist") else None
            result["explicit_indian_content"] = bool(result.get("explicit_indian_content")) or bool(re.search(r"\b(india|indian|hindi|tamil|telugu|malayalam|bengali|kannada|marathi|punjabi|bollywood)\b", question, re.IGNORECASE))
            result["language"] = _detect_media_language(question, result["artist"], result.get("language"))
            if result["religion"] == "Buddhism":
                result["category"] = "Buddhist Dhamma"
                result["youtube_search_query"] = "ဗုဒ္ဓဘာသာ တရားတော် မေတ္တာ စိတ်ငြိမ်းချမ်းမှု"
            elif result["religion"] in RELIGIOUS_QUERIES:
                result["youtube_search_query"] = RELIGIOUS_QUERIES[result["religion"]]
            result["max_results"] = 6
            result["youtube_search_query"] = str(result.get("youtube_search_query", "")).strip()
            if not result["youtube_search_query"]:
                return _fallback_media_request(question)
            if result["language"] == "en" and not result["explicit_indian_content"]:
                result["youtube_search_query"] += " -India -Hindi -Tamil -Telugu -Bollywood"
            result["source"] = question
            return result
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
        pass
    return _fallback_media_request(question)


@st.cache_data(ttl=900, show_spinner=False)
def search_youtube_videos(query, category, religion, language, max_results=6, media_request=None):
    if not YOUTUBE_API_KEY:
        return []
    videos = []
    seen_ids = set()
    page_token = None
    try:
        for _ in range(3):
            search_params = {"part": "snippet", "q": query, "type": "video", "maxResults": min(max(max_results * 4, 25), 50), "videoEmbeddable": "true", "safeSearch": "strict", "relevanceLanguage": language, "key": YOUTUBE_API_KEY}
            if page_token:
                search_params["pageToken"] = page_token
            search = requests.get("https://www.googleapis.com/youtube/v3/search", params=search_params, timeout=10)
            search.raise_for_status()
            search_data = search.json()
            ids = [item.get("id", {}).get("videoId") for item in search_data.get("items", []) if item.get("id", {}).get("videoId")]
            ids = [video_id for video_id in ids if video_id not in seen_ids]
            seen_ids.update(ids)
            if ids:
                details = requests.get("https://www.googleapis.com/youtube/v3/videos", params={"part": "snippet,contentDetails,status", "id": ",".join(ids), "key": YOUTUBE_API_KEY}, timeout=10)
                details.raise_for_status()
                for item in details.json().get("items", []):
                    video_id = str(item.get("id", "")).strip()
                    snippet = item.get("snippet", {}) if isinstance(item.get("snippet"), dict) else {}
                    embeddable = item.get("status", {}).get("embeddable", False)
                    embed_url = f"https://www.youtube.com/embed/{video_id}" if re.fullmatch(r"[A-Za-z0-9_-]{6,}", video_id) else ""
                    logger.info("YouTube candidate id=%r title=%r embeddable=%r embed_url=%r", video_id, snippet.get("title", ""), embeddable, embed_url)
                    if not embeddable or not embed_url:
                        logger.info("YouTube candidate rejected id=%r", video_id)
                        continue
                    if media_request and not is_safe_media_result(item, media_request):
                        logger.info("YouTube candidate rejected by safety/language filter id=%r", video_id)
                        continue
                    thumb = snippet.get("thumbnails", {}).get("high", snippet.get("thumbnails", {}).get("default", {})).get("url", "")
                    videos.append({"video_id": video_id, "title": snippet.get("title", "Untitled video"), "description": snippet.get("description", ""), "channel": snippet.get("channelTitle", "YouTube"), "thumbnail": thumb, "duration": item.get("contentDetails", {}).get("duration", ""), "published_date": snippet.get("publishedAt", ""), "watch_url": f"https://www.youtube.com/watch?v={video_id}", "embed_url": embed_url, "embeddable": True, "category": category, "religion": religion, "artist": media_request.get("artist") if media_request else None, "language": language})
                    logger.info("YouTube candidate accepted id=%s", video_id)
                    if len(videos) >= max_results:
                        return videos[:max_results]
            page_token = search_data.get("nextPageToken")
            if not page_token:
                break
        return videos
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return videos


def _format_duration(duration):
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return "Unknown"
    hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def render_assistant_response(response):
    """Render normal assistant text only; media is fetched separately."""
    response = _sanitize_assistant_response(response)
    if response:
        st.markdown(response)


def render_media_section(media_request, llm):
    """Fetch and render media without parsing URLs from the assistant response."""
    if not media_request.get("is_requested"):
        return
    if contains_sensitive_topics(media_request.get("source", "")):
        return
    if not YOUTUBE_API_KEY:
        st.info("Video recommendations are temporarily unavailable.")
        return
    videos = search_youtube_videos(
        media_request["youtube_search_query"], media_request["category"],
        media_request.get("religion"), media_request["language"], 6, media_request)
    if not videos:
        st.info("No suitable videos were found for this request.")
        return
    st.markdown("**Recommended Videos**")
    render_video_cards_from_list(videos, media_request["category"], media_request["language"])


def _media_assistant_response(media_request):
    artist = media_request.get("artist")
    if artist:
        return f"Listening to {artist} can be a positive way to support your mood. I'll show suitable songs below when they are available."
    return "Music and videos can be a supportive part of your wellbeing routine. I'll show suitable recommendations below when they are available."


# -------------------------------------------------------------------
# BOOK RESPONSE – uses fixed lists, no API calls
# -------------------------------------------------------------------

def _book_assistant_response(book_request, books, llm):
    language = "Burmese" if book_request.get("language") == "🇲🇲 Burmese" else "English"
    metadata = json.dumps([
        {key: book.get(key) for key in (
            "title", "authors", "description", "language", "category", "source",
            "type", "book_url", "resource_url", "read_url", "download_url",
            "format", "access_type", "availability", "readable", "is_free",
        )}
        for book in books[:BOOK_SEARCH_LIMIT]
    ], ensure_ascii=False)

    # Force correct Burmese disclaimer
    opening_instruction = (
        "Open with a short natural introduction stating that the resources are educational "
        "and not a substitute for professional mental-health care."
    )
    if language == "Burmese":
        opening_instruction += (
            " For Burmese, use exactly this disclaimer: "
            "'ဤစာရင်းထဲက အရင်းအမြစ်များသည် စိတ်ကျန်းမာရေးဆိုင်ရာ ပညာပေးအကြောင်းအရာများဖြစ်ပြီး၊ "
            "ပရော်ဖက်ရှင်နယ် စိတ်ကျန်းမာရေးစောင့်ရှောက်မှုအတွက် အစားထိုးမဟုတ်ပါ။'"
        )
    

    prompt = f"""Write a concise, friendly {language} introduction to the supplied BOOK_SEARCH_RESULTS.
Do not repeat all book metadata – the application separately renders the supplied books as visual book cards.
Do not generate HTML, CSS, SVG, Streamlit code, or card markup – return only plain Markdown text.
Use at most 6 numbered results, and include only supplied results with a meaningful title.
{opening_instruction}
For each result, include its supplied title, author, category, actual language, format, access status, concise supplied description, and source.
Include a Markdown Read/View link only when a supplied URL is HTTPS and non-empty; use read_url or download_url when supplied, otherwise book_url or resource_url. Never create or modify a URL and never use localhost.
Use FREE only when supplied is_free is true or supplied access_type/availability explicitly says FREE. Say Preview or View details for reference-only items.
If download_url is supplied and HTTPS, include it as a separate PDF/download link; otherwise do not mention or invent a PDF link.
Keep the opening and each item concise.

BOOK_SEARCH_RESULTS:
{metadata}

If the language is Burmese, write in natural Burmese Unicode. Mention that these are educational resources, not a replacement for professional care.
Use the actual language shown in each supplied result; never label an English result as Burmese."""
    try:
        response = llm.invoke(prompt, max_tokens=3000)
        text = _sanitize_assistant_response(getattr(response, "content", response))
        text = re.sub(r"```(?:html|xml|svg|css|python)?\s*|```", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?(?:article|div|section|style|script|a|img|button|span|p|h[1-6])\b[^>]*>", "", text, flags=re.IGNORECASE)
        return _sanitize_assistant_response(text)
    except Exception:
        if language == "Burmese":
            return "စစ်ဆေးပြီးသော စိတ်ကျန်းမာရေးစာအုပ်များနှင့် အရင်းအမြစ်များကို အောက်တွင် ဖော်ပြထားပါသည်။ ပညာပေးအထောက်အကူအဖြစ် အသုံးပြုပါ။"
        return "Here are verified mental-health books and resources for educational support. They are not a replacement for professional care."


def detect_book_request(question):
    lowered = (question or "").casefold()
    has_book_term = any(term in lowered or term in question for term in BOOK_REQUEST_TERMS)
    has_book_intent = bool(re.search(
        r"\b(recommend|suggest|find|search|show|give|need|looking for|want|provide)\b|"
        r"စာအုပ်.*(ရှာ|အကြံပြု|ပေး)|"
        r"(ရှာ|အကြံပြု|ပေး).*စာအုပ်",
        question or "",
        re.IGNORECASE,
    ))
    has_mental_topic = any(
        term.casefold() in lowered
        for terms in BOOK_TOPIC_TERMS.values()
        for term in terms
    )
    requested = has_book_term and (has_book_intent or has_mental_topic)
    is_burmese = _is_burmese_request(question)
    requested_limit = re.search(r"\b([1-9]|10)\s*(?:books?|စာအုပ်)", question, re.IGNORECASE)
    limit = min(int(requested_limit.group(1)), BOOK_SEARCH_LIMIT) if requested_limit else BOOK_SEARCH_LIMIT
    topic = next(
        (name for name, terms in BOOK_TOPIC_TERMS.items() if any(term.casefold() in lowered for term in terms)),
        "mental health",
    )
    return {
        "is_requested": requested,
        "language": "🇲🇲 Burmese" if is_burmese else "English",
        "topic": topic,
        "free_only": bool(re.search(r"\bfree\b|\bfreely\b|\bread online\b|အခမဲ့|အခမဲ့ဖတ်|အခမဲ့ဒေါင်း", lowered)),
        "limit": limit,
        "source": question,
    }


# -------------------------------------------------------------------
# These functions are kept only for fallback – not used in the fixed flow
# -------------------------------------------------------------------
def _official_resource_is_available(resource_url):
    try:
        response = requests.head(resource_url, allow_redirects=True, timeout=10)
        if response.status_code in (403, 405) or response.status_code >= 400:
            response = requests.get(resource_url, allow_redirects=True, timeout=10, stream=True)
        return response.status_code < 400 and urlparse(response.url).hostname in {"mhpsshub.org", "www.mhpsshub.org"}
    except requests.RequestException:
        return False


def _trusted_resource_is_available(resource_url):
    trusted_hosts = {
        "who.int", "www.who.int", "iris.who.int", "openlibrary.org", "archive.org",
        "www.archive.org", "gutenberg.org", "www.gutenberg.org",
    }
    try:
        response = requests.get(resource_url, allow_redirects=True, timeout=10, stream=True)
        hostname = (urlparse(response.url).hostname or "").lower()
        return response.status_code < 400 and any(
            hostname == host or hostname.endswith(f".{host}") for host in trusted_hosts
        )
    except requests.RequestException:
        return False


class _PdfLinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.pdf_links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        if ".pdf" in href.lower():
            self.pdf_links.append(href)


def _official_pdf_url(resource_url):
    try:
        response = requests.get(resource_url, timeout=10)
        response.raise_for_status()
        if "application/pdf" in response.headers.get("content-type", "").lower():
            return response.url
        parser = _PdfLinkParser()
        parser.feed(response.text)
        for link in parser.pdf_links:
            pdf_url = urljoin(response.url, link)
            if urlparse(pdf_url).hostname in {"mhpsshub.org", "www.mhpsshub.org"}:
                return pdf_url
    except (requests.RequestException, ValueError):
        return ""
    return ""


def _trusted_pdf_url(resource_url):
    trusted_hosts = {"who.int", "www.who.int", "iris.who.int", "openlibrary.org", "archive.org", "www.archive.org", "gutenberg.org", "www.gutenberg.org"}
    try:
        response = requests.get(resource_url, timeout=10)
        response.raise_for_status()
        if "application/pdf" in response.headers.get("content-type", "").lower():
            return response.url
        parser = _PdfLinkParser()
        parser.feed(response.text)
        for link in parser.pdf_links:
            pdf_url = urljoin(response.url, link)
            hostname = (urlparse(pdf_url).hostname or "").lower()
            if any(hostname == host or hostname.endswith(f".{host}") for host in trusted_hosts):
                return pdf_url
    except (requests.RequestException, ValueError):
        return ""
    return ""


def _extract_saved_cover(resource):
    """Render and cache the first page of an official MHPSS Hub PDF."""
    catalog_cover = str(resource.get("cover_url", ""))
    if catalog_cover.startswith("/") and not catalog_cover.startswith("//"):
        catalog_path = os.path.join(working_dir, catalog_cover.lstrip("/"))
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "rb") as cover_file:
                    encoded = base64.b64encode(cover_file.read()).decode("ascii")
                media_type = mimetypes.guess_type(catalog_path)[0] or "image/jpeg"
                return f"data:{media_type};base64," + encoded
            except OSError:
                logger.warning("Could not read cached catalog cover: %s", catalog_path)
    try:
        fitz = importlib.import_module("fitz")
    except (ImportError, ModuleNotFoundError):
        logger.warning("PyMuPDF is not installed; using fallback cover for %s", resource["title"])
        return ""
    pdf_url = _official_pdf_url(resource["resource_url"])
    if not pdf_url:
        return ""
    filename = hashlib.sha256(resource["resource_url"].encode("utf-8")).hexdigest() + ".jpg"
    cover_path = os.path.join(BOOK_COVER_DIR, filename)
    try:
        os.makedirs(BOOK_COVER_DIR, exist_ok=True)
        if not os.path.exists(cover_path):
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            document = fitz.open(stream=response.content, filetype="pdf")
            if not document.page_count:
                document.close()
                return ""
            document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(cover_path)
            document.close()
        with open(cover_path, "rb") as cover_file:
            encoded = base64.b64encode(cover_file.read()).decode("ascii")
        return "data:image/jpeg;base64," + encoded
    except (OSError, ValueError, RuntimeError, requests.RequestException):
        logger.exception("Could not create PDF cover for %s", resource["title"])
        return ""


def _fallback_cover_data_uri(title):
    safe_title = escape(str(title), quote=True).replace("\n", " ")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800" viewBox="0 0 600 800"><rect width="600" height="800" fill="#dceae4"/><rect x="34" y="34" width="532" height="732" rx="18" fill="#b8d2c7" stroke="#24504b" stroke-width="4"/><text x="300" y="155" fill="#16746b" font-family="Arial,sans-serif" font-size="26" font-weight="700" text-anchor="middle">MHPSS HUB</text><text x="300" y="350" fill="#163b39" font-family="Arial,sans-serif" font-size="30" font-weight="700" text-anchor="middle"><tspan x="300" dy="0">{safe_title[:34]}</tspan><tspan x="300" dy="44">{safe_title[34:68]}</tspan><tspan x="300" dy="44">{safe_title[68:102]}</tspan></text><text x="300" y="680" fill="#24504b" font-family="Arial,sans-serif" font-size="22" text-anchor="middle">Burmese Mental Health Resource</text></svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def deduplicate_books(books):
    seen_ids = set()
    seen_titles = set()
    result = []
    for book in books:
        book_id = str(book.get("id", "")).strip()
        title = re.sub(r"\s+", " ", str(book.get("title", "")).casefold()).strip()
        authors = book.get("authors", book.get("author", []))
        if isinstance(authors, str):
            authors = [authors]
        author_key = ",".join(str(author).casefold().strip() for author in authors)
        title_key = f"{title}|{author_key}"
        if (book_id and book_id in seen_ids) or title_key in seen_titles:
            continue
        if book_id:
            seen_ids.add(book_id)
        seen_titles.add(title_key)
        result.append(book)
    return result


def _book_topic(query):
    lowered = str(query or "").casefold()
    return next(
        (name for name, terms in BOOK_TOPIC_TERMS.items()
         if any(term.casefold() in lowered for term in terms)),
        "mental health",
    )


def _book_query_variants(query, language):
    topic = next((name for name, terms in BOOK_TOPIC_TERMS.items()
                  if any(term.casefold() in query.casefold() for term in terms)), "mental health")
    english_topic = next((term for term in BOOK_TOPIC_TERMS[topic] if term.isascii()), topic)
    if language == "🇲🇲 Burmese":
        burmese_terms = " ".join(term for term in BOOK_TOPIC_TERMS[topic] if not term.isascii()) or query
        variants = [
            f"{burmese_terms} စိတ်ကျန်းမာရေး", f"{burmese_terms} စိတ်ပညာ",
            f"{burmese_terms} စိတ်ပူပန်မှု", f"{burmese_terms} စိတ်ဖိစီးမှု",
            "စိတ်ကျန်းမာရေး", "စိတ်ပညာ", "စိတ်ဖိစီးမှု", "စိတ်ပူပန်မှု",
            "Burmese mental health", "Burmese psychology", "Burmese anxiety", "Burmese stress",
        ]
    else:
        variants = [
            f"{english_topic} mental health", f"{english_topic} psychology",
            f"{english_topic} anxiety", f"{english_topic} stress",
            f"{english_topic} mindfulness", "mental health self help",
        ]
    return [f"{variant} -subject:report -subject:government" for variant in variants]


def is_free_readable_google_book(volume):
    access = volume.get("accessInfo", {}) if isinstance(volume.get("accessInfo"), dict) else {}
    viewability = str(access.get("viewability", "")).upper()
    access_view_status = str(access.get("accessViewStatus", "")).upper()
    public_domain = bool(access.get("publicDomain", False))
    epub = access.get("epub", {}) or {}
    pdf = access.get("pdf", {}) or {}
    epub_available = bool(epub.get("isAvailable", False)) if isinstance(epub, dict) else False
    pdf_available = bool(pdf.get("isAvailable", False)) if isinstance(pdf, dict) else False
    web_reader = str(access.get("webReaderLink", "")).strip()

    readable_link = urlparse(web_reader)
    if readable_link.scheme != "https" or not readable_link.netloc:
        return False
    if public_domain and web_reader:
        return True
    if viewability == "ALL_PAGES" and web_reader:
        return True
    if access_view_status == "FULL_PUBLIC_DOMAIN" and web_reader:
        return True
    if epub_available and web_reader:
        return True
    if pdf_available and web_reader:
        return True
    return False


def _verified_https_url(value):
    parsed = urlparse(str(value or "").strip())
    return str(value).strip() if parsed.scheme == "https" and parsed.netloc else ""


def extract_cover_url(volume_info):
    image_links = volume_info.get("imageLinks", {}) if isinstance(volume_info, dict) else {}
    for size in ("extraLarge", "large", "medium", "small", "thumbnail", "smallThumbnail"):
        raw_cover = str(image_links.get(size) or "").strip().replace("http://", "https://", 1)
        cover = _verified_https_url(raw_cover)
        if cover:
            return cover
    return ""


def _normalize_google_book(volume, query, requested_language):
    info = volume.get("volumeInfo", {}) if isinstance(volume.get("volumeInfo"), dict) else {}
    access = volume.get("accessInfo", {}) if isinstance(volume.get("accessInfo"), dict) else {}
    epub = access.get("epub", {}) if isinstance(access.get("epub"), dict) else {}
    pdf = access.get("pdf", {}) if isinstance(access.get("pdf"), dict) else {}
    is_free = is_free_readable_google_book(volume)
    read_url = _verified_https_url(access.get("webReaderLink"))
    preview_url = _verified_https_url(info.get("previewLink") or info.get("infoLink"))
    download_url = _verified_https_url(pdf.get("downloadLink") or epub.get("downloadLink"))
    if not is_free:
        download_url = ""
    authors = [str(author).strip() for author in info.get("authors", []) if str(author).strip()]
    description = re.sub(r"<[^>]+>", " ", str(info.get("description", "")))
    description = re.sub(r"\s+", " ", description).strip()
    cover_url = extract_cover_url(info)
    category = ", ".join(map(str, info.get("categories", []))) or "Mental Health"
    return {
        "title": str(info.get("title", "Untitled book")).strip(),
        "authors": authors or ["Author unavailable"],
        "author": ", ".join(authors) if authors else None,
        "description": description or "Description unavailable.",
        "language": requested_language,
        "category": category,
        "topic": _book_topic(query),
        "type": "Guide" if "guide" in str(info.get("title", "")).casefold() else "Book",
        "cover_url": cover_url,
        "book_url": preview_url,
        "resource_url": read_url or preview_url,
        "read_url": read_url,
        "download_url": download_url,
        "source": "Google Books",
        "access_type": "FREE" if is_free else "REFERENCE",
        "readable": is_free,
        "is_free": is_free,
        "format": "PDF" if download_url else "Online book",
        "availability": "FREE" if is_free else ("PREVIEW" if preview_url else "REFERENCE"),
        "score": _book_relevance_score(volume, query),
        "id": str(volume.get("id", "")).strip(),
    }


def _book_relevance_score(volume, query):
    info = volume.get("volumeInfo", {}) if isinstance(volume.get("volumeInfo"), dict) else {}
    title = f"{info.get('title', '')} {info.get('subtitle', '')}".casefold()
    description = str(info.get("description", "")).casefold()
    categories = " ".join(map(str, info.get("categories", []))).casefold()
    authors = " ".join(map(str, info.get("authors", []))).casefold()
    terms = BOOK_MENTAL_TERMS
    score = 0
    topic = next((name for name, values in BOOK_TOPIC_TERMS.items() if any(value.casefold() in query.casefold() for value in values)), "mental health")
    topic_terms = BOOK_TOPIC_TERMS.get(topic, [])
    if any(term.casefold() in title for term in topic_terms):
        score += 10
    if any(term in title for term in terms):
        score += 8
    if any(term in description for term in terms):
        score += 6
    if any(term in categories for term in terms):
        score += 5
    if any(term.casefold() in description for term in topic_terms):
        score += 6
    if "psycholog" in authors or "psychiatr" in authors:
        score += 2
    return score


@st.cache_data(ttl=900, show_spinner=False)
def search_google_books(query, language="English", max_results=BOOK_SEARCH_LIMIT, free_only=False):
    # Not used in fixed flow, but kept for compatibility
    return []


def search_verified_free_resources(query, language, max_results=BOOK_SEARCH_LIMIT):
    # Not used in fixed flow, but kept for compatibility
    return []


@st.cache_data(ttl=900, show_spinner=False)
def search_mental_health_books(query, language, max_results=BOOK_SEARCH_LIMIT, free_only=False):
    """
    Returns the fixed list of 6 books based on the requested language.
    """
    if language == "🇲🇲 Burmese":
        return FIXED_BURMESE_BOOKS[:BOOK_SEARCH_LIMIT]
    else:
        return FIXED_ENGLISH_BOOKS[:BOOK_SEARCH_LIMIT]


# -------------------------------------------------------------------
# BOOK CARD RENDERER – COMPACT, FIXED SIZE
# -------------------------------------------------------------------
def render_book_cards(books):
    """Render compact book cards with vertical 2:3 cover aspect ratio."""
    if not books:
        return
    cards = []
    for book in books[:BOOK_SEARCH_LIMIT]:
        title_raw = str(book.get("title", "Untitled book")).strip()
        title = escape(title_raw, quote=True)
        author = book.get("authors", book.get("author", []))
        if isinstance(author, list):
            author = ", ".join(str(item).strip() for item in author if str(item).strip())
        author_text = escape(str(author).strip() or "Author unavailable", quote=True)
        description = escape(str(book.get("description", "Description unavailable.")).strip()[:100] or "Description unavailable.", quote=True)
        language = escape(str(book.get("language", "English")), quote=True)
        
        # ---------- Handle cover image ----------
        cover_url = str(book.get("cover_url", "")).strip()
        
        # If it's a local file path (not http/data)
        if cover_url and not cover_url.startswith(("http", "data:")):
            # If cover_url is already an absolute path, use it directly
            if os.path.isabs(cover_url):
                local_path = cover_url
            else:
                # Otherwise, join with the local image folder
                local_path = os.path.join(LOCAL_IMAGE_FOLDER, cover_url)
            
            if os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        img_data = f.read()
                    encoded = base64.b64encode(img_data).decode("ascii")
                    mime_type = mimetypes.guess_type(local_path)[0] or "image/jpeg"
                    cover_url = f"data:{mime_type};base64,{encoded}"
                except Exception:
                    # On error, fallback to default cover
                    cover_url = _fallback_cover_data_uri(title_raw)
            else:
                # File not found, fallback to default cover
                cover_url = _fallback_cover_data_uri(title_raw)
        elif not cover_url:
            cover_url = _fallback_cover_data_uri(title_raw)
        # ---------------------------------------

        action_url = str(book.get("read_url") or book.get("resource_url") or "").strip()
        parsed_link = urlparse(action_url)
        safe_link = escape(action_url, quote=True) if parsed_link.scheme == "https" and parsed_link.netloc else ""
        action_label = "Read Book" if book.get("readable") is True else "Preview"
        card_button = (
            f'<a class="book-read-button" href="{safe_link}" target="_blank" rel="noopener noreferrer">'
            f'<span class="watch-icon">&#128214;</span> {action_label}</a>'
            if safe_link else ""
        )
        cards.append(
            f'<article class="book-card-item"><div class="book-cover-wrap">'
            f'<img class="book-cover" src="{escape(cover_url, quote=True)}" alt="{title}" loading="lazy" referrerpolicy="no-referrer" crossorigin="anonymous">'
            f'<span class="book-free-badge">{escape(str(book.get("availability", "REFERENCE")))}</span>'
            f'</div><div class="book-card-content"><h4 class="book-title" title="{title}">{title}</h4>'
            f'<p class="book-author" title="{author_text}">{author_text}</p>'
            f'<p class="book-description">{description}</p>'
            f'<div class="book-meta"><span>{language}</span><span>{escape(str(book.get("category", "Mental Health")))}</span>'
            f'<span>{escape(str(book.get("source", "Verified source")))}</span></div>'
            f'{card_button}</div></article>'
        )
    st.markdown("""
    <style>
        .google-book-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; width: 100%; max-width: 1000px; margin: 0 auto 12px; align-items: stretch; }
        .book-card-item { display: flex; height: 100%; flex-direction: column; gap: 4px; width: 100%; min-width: 0; }
        .book-cover-wrap { 
            position: relative; 
            width: 100%; 
            aspect-ratio: 2 / 3;  /* Vertical 2:3 aspect ratio */ 
            overflow: hidden; 
            border: 1px solid rgba(15, 23, 42, .08); 
            border-radius: 8px; 
            background: #f0f0f0; 
            box-shadow: 0 2px 6px rgba(15, 23, 42, .05); 
            transition: all .25s ease; 
        }
        .book-cover-wrap:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15, 23, 42, .1); }
        .book-cover { 
            display: block; 
            width: 100%; 
            height: 100%; 
            object-fit: contain;  /* Show full cover without cropping */
            background: #f0f0f0; 
        }
        .book-free-badge { position: absolute; top: 6px; left: 6px; padding: 2px 6px; border-radius: 6px; background: rgba(22, 116, 107, .9); color: #fff; font-size: .5rem; font-weight: 700; }
        .book-card-content { display: flex; flex: 1; flex-direction: column; gap: 2px; justify-content: space-between; padding: 6px 2px 2px 2px; }
        .book-title { display: -webkit-box; min-height: 2.2em; margin: 0; overflow: hidden; color: #0f172a; font-size: .72rem; line-height: 1.2; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
        .book-author { margin: 0; overflow: hidden; color: #475569; font-size: .6rem; line-height: 1.2; text-overflow: ellipsis; white-space: nowrap; }
        .book-description { display: -webkit-box; margin: 0; overflow: hidden; color: #64748b; font-size: .58rem; line-height: 1.2; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
        .book-meta { display: flex; flex-wrap: wrap; gap: 3px; color: #64748b; font-size: .5rem; }
        .book-meta span { max-width: 100%; overflow: hidden; padding: 1px 4px; border-radius: 6px; background: #f1f5f9; text-overflow: ellipsis; white-space: nowrap; }
        .book-read-button { display: flex; align-items: center; justify-content: center; min-height: 24px; margin-top: auto; border-radius: 6px; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff !important; font-size: .65rem; font-weight: 600; line-height: 1.2; text-decoration: none !important; transition: all .15s ease; box-shadow: 0 2px 4px rgba(37, 99, 235, .15); }
        .book-read-button:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af); transform: translateY(-1px); box-shadow: 0 4px 8px rgba(37, 99, 235, .25); }
        @media (max-width: 850px) { .google-book-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); max-width: 700px; } }
        @media (max-width: 550px) { .google-book-grid { grid-template-columns: 1fr; max-width: 400px; } }
    </style>
    <div class="google-book-grid">""" + "".join(cards) + "</div>", unsafe_allow_html=True)


def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

class SimpleChatChain:
    def __init__(self, vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
        self.vectorstore = vectorstore
        self.system_prompt = system_prompt
        self.negative_prompt = negative_prompt
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=MAX_OUTPUT_TOKENS)

    @staticmethod
    def _prompt_tokens(prompt):
        return max(1, (len(prompt) + 3) // 4)

    @staticmethod
    def _is_request_too_large(error):
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status_code = status_code or getattr(response, "status_code", None)
        error_text = str(error).lower()
        return status_code == 413 or "request too large" in error_text or "tokens per minute" in error_text

    def _build_prompt(self, question, context, chat_history, context_chars, history_chars):
        context_text = context[:context_chars]
        language_instruction = _conversation_language_instruction(chat_history)
        system_prompt = TEXT_RESPONSE_SYSTEM_PROMPT
        if language_instruction:
            system_prompt = f"{system_prompt}\n\nCURRENT LANGUAGE INSTRUCTION:\n{language_instruction}"
        history_text = "\n".join(
            f"{entry.get('role', 'user').capitalize()}: {str(entry.get('content', ''))[:2000]}"
            for entry in chat_history[-MAX_HISTORY_MESSAGES:]
            if isinstance(entry, dict)
        ) or "No previous conversation."
        history_text = history_text[-history_chars:]
        question_text = question[:4000]
        prompt = PromptTemplate.from_template(
            """{system_prompt}

Context:
{context}

Recent conversation:
{chat_history}

User question: {question}

Answer:"""
        ).format(
            system_prompt=system_prompt,
            context=context_text or "No relevant mental-health context found.",
            chat_history=history_text,
            question=question_text,
        )
        while self._prompt_tokens(prompt) > TARGET_PROMPT_TOKENS and (context_text or history_text):
            if len(context_text) >= len(history_text) and context_text:
                context_text = context_text[:max(0, len(context_text) - 1000)]
            elif history_text:
                history_text = history_text[-max(0, len(history_text) - 1000):]
            prompt = PromptTemplate.from_template(
                """{system_prompt}

Context:
{context}

Recent conversation:
{chat_history}

User question: {question}

Answer:"""
            ).format(
                system_prompt=system_prompt,
                context=context_text or "No relevant mental-health context found.",
                chat_history=history_text,
                question=question_text,
            )
        return prompt, len(context_text), len([entry for entry in chat_history[-MAX_HISTORY_MESSAGES:] if isinstance(entry, dict)])

    def __call__(self, inputs):
        question = str(inputs.get("question", "")).strip()
        chat_history = inputs.get("chat_history", [])
        if not question:
            return {"answer": "Please ask a question about mental health."}

        docs = self.vectorstore.similarity_search(question, k=RETRIEVAL_K)
        context = "\n\n".join(
            str(doc.page_content)[:MAX_CONTEXT_CHARS // RETRIEVAL_K]
            for doc in docs
            if getattr(doc, "page_content", None)
        )
        if not context:
            context = "No relevant mental-health context found in the database."

        formatted_prompt, context_length, history_count = self._build_prompt(
            question, context, chat_history, MAX_CONTEXT_CHARS, MAX_HISTORY_CHARS
        )
        prompt_tokens = self._prompt_tokens(formatted_prompt)
        logger.info(
            "RAG prompt chars=%d approx_tokens=%d retrieved_docs=%d history_messages=%d",
            len(formatted_prompt), prompt_tokens, len(docs), history_count,
        )
        if DEBUG_PROMPT_SIZE:
            st.caption(f"Prompt: {len(formatted_prompt):,} chars (~{prompt_tokens:,} tokens); retrieved documents: {len(docs)}; history messages: {history_count}")

        try:
            response = self.llm.invoke(formatted_prompt)
        except Exception as error:
            if not self._is_request_too_large(error):
                logger.exception("LLM request failed")
                return {"answer": "I couldn't generate a response right now. Please try again.", "books": [], "videos": []}
            logger.warning("LLM request was too large; retrying with reduced context and history")
            formatted_prompt, context_length, history_count = self._build_prompt(
                question, context, chat_history, RETRY_CONTEXT_CHARS, RETRY_HISTORY_CHARS
            )
            retry_tokens = self._prompt_tokens(formatted_prompt)
            logger.info(
                "RAG retry prompt chars=%d approx_tokens=%d retrieved_docs=%d history_messages=%d",
                len(formatted_prompt), retry_tokens, len(docs), history_count,
            )
            try:
                response = self.llm.invoke(formatted_prompt)
            except Exception as retry_error:
                logger.exception("LLM retry failed")
                return {"answer": "The request was too large to process. Please try a shorter question.", "books": [], "videos": []}
        answer = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer, "books": [], "videos": []}


def chat_chain(vectorstore, system_prompt=DEFAULT_SYSTEM_PROMPT, negative_prompt=DEFAULT_NEGATIVE_PROMPT):
    return SimpleChatChain(vectorstore, system_prompt, negative_prompt)

st.set_page_config(
    page_title="Chat with Genesis Care",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for sidebar styling
st.markdown("""
    <style>
    div.css-textbarboxtype {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    
    /* Justify text for Purpose section */
    div.css-textbarboxtype:nth-of-type(3) {
        text-align: justify;
        text-justify: inter-word;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About Bot")
    
    # About Section
    st.markdown("## Description")
    st.markdown("""
        <div class="css-textbarboxtype">
            An AI-powered chatbot designed to provide mental health support.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Goals")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Provide 24/7 mental health support<br>
            - Offer crisis intervention when needed<br>
            - Connect users with professional resources
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Purpose")
    st.markdown("""
        <div class="css-textbarboxtype">
            Designed as a stigma-free entry point to mental health support in Myanmar—where people often feel too shy to seek traditional therapy—this chatbot tackles interpersonal harassment, workplace stress, and suicidal ideation by offering empathetic listening, practical coping strategies, and evidence-based guidance from the WHO and Myanmar's Ministry of Health and Family Welfare. By normalizing conversations about emotional well-being and delivering timely, trustworthy advice, it bridges users to professional therapists when they're ready.
        </div>
    """, unsafe_allow_html=True)
    
    # Values
    st.markdown("## Our Values")
    st.markdown("""
        <div class="css-textbarboxtype">
            - Empathy<br>
            - Professional Ethics<br>
            - User Safety
        </div>
    """, unsafe_allow_html=True)
    
    # Chat History Section
    st.markdown("---")
    st.markdown("## Chat History")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history previews
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            if st.button(f"Chat {idx//2 + 1}: {message['content'][:30]}...", key=f"history_{idx}"):
                # Load this conversation
                st.session_state.selected_chat = idx//2
    
    # PDF Export Button
    st.markdown("---")
    if st.button("Export Chat to PDF"):
        if len(st.session_state.chat_history) > 0:
            try:
                pdf_data = build_chat_history_pdf(st.session_state.chat_history)
                filename = f"mental_health_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf"
                )
                
            except Exception as e:
                logger.exception("Could not export chat history as PDF")
                st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("No chat history to export!")

# Main chat interface
# Add header image

import base64
import streamlit as st

# Function to encode the image file to base64 for HTML embedding
image_path = (
    "/Users/daniel/Downloads/Old/images/telegram-cloud-photo-size-5-6079859858187424447-x.jpg"
)

with open(image_path, "rb") as f:
    img_bytes = f.read()
encoded_img = base64.b64encode(img_bytes).decode()

# Inline flexbox container to lock icon and header side-by-side
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <img src="data:image/jpeg;base64,{encoded_img}" style="width: 55px; height: 55px; border-radius: 50%; object-fit: cover;">
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; line-height: 1.2;">Genesis Care</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

image_candidates = [
    os.path.join(working_dir, "images", "Panromic_May_Is_Mental_HEALTH_Awareness_Month_image.png"),
    os.path.join(working_dir, "images", "banner-for-mental-health-awareness-month-in-may.jpg"),
    os.path.join(working_dir, "images", "may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector-2.jpg"),
    os.path.join(working_dir, "images", "may-is-mental-health-awareness-month-diversity-silhouettes-of-adults-and-children-of-different-nationalities-and-appearances-colorful-people-contour-in-flat-style-vector.jpg"),
]
selected_image = next((path for path in image_candidates if os.path.exists(path)), None)
if selected_image:
    st.image(selected_image, use_column_width="auto")

if not GROQ_API_KEY:
    st.warning("Groq API key is missing. Add it to config.json or set the GROQ_API_KEY environment variable to enable chatbot responses.")
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Display chat messages – now including book cards for stability on reruns
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_response(message["content"])
            if message.get("media_request"):
                render_media_section(message["media_request"], st.session_state.conversational_chain.llm)
            # Only show book cards if the request was for English books
            if message.get("books") and message.get("book_request", {}).get("language") != "🇲🇲 Burmese":
                render_book_cards(message["books"])
        else:
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about Mental Health")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        media_request = {"is_requested": False}
        book_request = detect_book_request(user_input) if not is_translation_request(user_input) else {"is_requested": False}
        books = []
        if is_translation_request(user_input):
            assistant_response = translate_previous_response(
                user_input,
                st.session_state.chat_history,
                st.session_state.conversational_chain.llm,
            )
        else:
            if not book_request.get("is_requested") and not contains_sensitive_topics(user_input):
                media_request = detect_media_request(user_input, st.session_state.conversational_chain.llm)
            if contains_sensitive_topics(user_input):
                response = st.session_state.conversational_chain({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history,
                })
                assistant_response = response["answer"]
            elif book_request.get("is_requested"):
                # Use the fixed book list based on language
                books = search_mental_health_books(
                    user_input, book_request.get("language", "English"),
                    book_request.get("limit", BOOK_SEARCH_LIMIT), book_request.get("free_only", False),
                )
                assistant_response = _book_assistant_response(
                    book_request, books, st.session_state.conversational_chain.llm
                )
                # Render book cards only for non‑Burmese resources
                if books and book_request.get("language") != "🇲🇲 Burmese":
                    render_book_cards(books)
                    logger.info("BOOK UI | requested=%s | found=%d", book_request.get("source"), len(books))
            elif media_request.get("is_requested"):
                assistant_response = _media_assistant_response(media_request)
            else:
                response = st.session_state.conversational_chain({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history,
                })
                assistant_response = response["answer"]
        assistant_response = _sanitize_assistant_response(assistant_response)
        render_assistant_response(assistant_response)
        if not is_translation_request(user_input) and not book_request.get("is_requested") and not contains_sensitive_topics(user_input) and not media_request.get("is_requested"):
            media_request = detect_media_request(user_input, st.session_state.conversational_chain.llm)
            render_media_section(media_request, st.session_state.conversational_chain.llm)
        elif media_request.get("is_requested"):
            render_media_section(media_request, st.session_state.conversational_chain.llm)
        # Store the message with book data for stability on reruns
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "media_request": media_request,
            "book_request": book_request,
            "books": books,
        })