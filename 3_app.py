import streamlit as st
import pandas as pd
import requests
import re
import os
import time
import base64
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PIL import Image

# --- 1. KONFIGURASI API KEY ---
API_KEY = "AIzaSyB5vKHle3HXD9EnZS_RCIQJki9jrzaHxyk"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-flash-latest')

# --- 2. SETUP HALAMAN & LOGO ---
try:
    logo_icon = Image.open("usi 2.png") 
except FileNotFoundError:
    logo_icon = "🤖"

st.set_page_config(
    page_title="USI (USMTV Assistant)",
    page_icon=logo_icon,
    layout="centered",
    initial_sidebar_state="auto"
)

# --- CSS FIX ---
st.markdown("""
    <style>
        .reportview-container { margin-top: -2em; }
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        [data-testid="stSidebarCollapsedControl"] {
            display: block !important;
            color: #d32f2f;
        }
        
        /* CSS Khusus untuk Logo di dalam teks */
        .inline-logo {
            vertical-align: middle;
            margin: 0 4px;
            transition: transform 0.2s;
            border-radius: 4px;
        }
        .inline-logo:hover {
            transform: scale(1.15); /* Membesar saat disentuh mouse */
        }
        
        /* CSS Kotak Info Biru */
        .custom-info-box {
            background-color: #eef4ff;
            padding: 12px 16px;
            border-radius: 8px;
            color: #004085;
            font-size: 14px;
            margin-bottom: 15px;
            border-left: 4px solid #3b82f6;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

ICON_USI = "usi 2.png" 
ICON_USER = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"

# --- 3. FUNGSI UPDATE DATA ---
def update_database_otomatis():
    url = "https://usmtv.id/wp-json/wp/v2/posts?per_page=15&_fields=title,link,content,date"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return False
        data_mentah = response.json()
    except Exception:
        return False
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def bersihkan(teks):
        teks = re.sub(r'<[^>]+>', ' ', str(teks))
        teks = re.sub(r'[^a-zA-Z0-9\s]', '', teks).lower()
        return stemmer.stem(teks)

    data_bersih = []
    for item in data_mentah:
        raw = item['content']['rendered']
        data_bersih.append({
            'judul': item['title']['rendered'],
            'link': item['link'],
            'isi_html': raw, 
            'teks_bersih': bersihkan(raw) 
        })

    pd.DataFrame(data_bersih).to_csv('dataset_bersih.csv', index=False)
    return True

# Cek & Auto-Update setiap 6 Jam
def cek_kesehatan_data():
    file_path = 'dataset_bersih.csv'
    perlu_update = False
    if not os.path.exists(file_path):
        perlu_update = True
    else:
        try:
            umur_file = time.time() - os.path.getmtime(file_path)
            if umur_file > 21600: 
                perlu_update = True
        except:
            perlu_update = True 
    
    if perlu_update:
        with st.spinner("Sedang sinkronisasi berita terbaru otomatis..."):
            update_database_otomatis()
            st.cache_resource.clear()

cek_kesehatan_data()

# --- 4. LOAD DATA ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('dataset_bersih.csv')
        df['teks_bersih'] = df['teks_bersih'].fillna('') 
    except FileNotFoundError:
        return None, None, None, None

    vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
    tfidf_matrix = vectorizer.fit_transform(df['teks_bersih'])
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return df, vectorizer, tfidf_matrix, stemmer

df, vectorizer, tfidf_matrix, stemmer = load_data()

# --- 5. LOGIKA USI ---
def tanya_usi(pertanyaan_user):
    if df is None: return "Database bermasalah."

    clean_query = stemmer.stem(pertanyaan_user.lower())
    query_vec = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][-3:][::-1]

    konteks_berita = ""
    for idx in top_indices:
        judul = df.iloc[idx]['judul']
        link = df.iloc[idx]['link']
        isi = df.iloc[idx]['isi_html']
        konteks_berita += f"JUDUL: {judul}\nISI: {isi}\nLINK SUMBER: {link}\n\n"

    prompt = f"""
    Kamu adalah USI, asisten USMTV.
    Jawab SINGKAT, PADAT, dan JELAS berdasarkan KONTEKS BERITA:
    {konteks_berita}
    
    ATURAN MUTLAK:
    Di akhir setiap jawaban, KAMU WAJIB menyertakan link sumber beritanya dengan format:
    "Sumber: [Link Berita]"
    
    Pertanyaan: {pertanyaan_user}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Maaf, koneksi gangguan."

# --- 6. SIDEBAR (HANYA PENGATURAN) ---
with st.sidebar:
    st.image(ICON_USI, width=80)
    
    st.markdown("<br>", unsafe_allow_html=True) 
    
    st.write("**Pengaturan**")
    
    if st.button("Update Berita"):
        with st.spinner("Updating..."):
            if update_database_otomatis():
                st.success("Updated!")
                st.cache_resource.clear() 

    if st.button("Hapus Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 7. TAMPILAN CHAT UTAMA & LOGO INLINE ---
st.title("USI Si Asisten!")

# Blok Info & Logo Kontributor (SEKARANG AKAN SELALU MUNCUL)
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

file_logo_email = "logo_email.png" 
file_logo_wa = "logo_wa.jpg"       

img_email = get_image_base64(file_logo_email)
img_wa = get_image_base64(file_logo_wa)

if img_email and img_wa:
    info_html = f"""
    <div class="custom-info-box">
        Ingin mengunggah berita anda sendiri? Hubungi (
        <a href="mailto:dyahretnosupriyani@gmail.com" target="_blank" title="Kirim Email">
            <img src="data:image/png;base64,{img_email}" height="22" class="inline-logo">
        </a> 
        ) atau WhatsApp (
        <a href="https://wa.me/62895411855225" target="_blank" title="Chat WhatsApp">
            <img src="data:image/png;base64,{img_wa}" height="22" class="inline-logo">
        </a> 
        ) kami.
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)
else:
    st.info("Ingin mengunggah berita anda sendiri? Hubungi Email (dyahretnosupriyani@gmail.com) atau WhatsApp (0895411855225) kami.")

# --- 8. AREA CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya USI. Ada berita yang dicari?"}]

for message in st.session_state.messages:
    avatar = ICON_USER if message["role"] == "user" else ICON_USI
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya berita..."):
    st.chat_message("user", avatar=ICON_USER).write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner('Mencari...'):
        jawaban_usi = tanya_usi(prompt)
    
    st.chat_message("assistant", avatar=ICON_USI).write(jawaban_usi)
    st.session_state.messages.append({"role": "assistant", "content": jawaban_usi})