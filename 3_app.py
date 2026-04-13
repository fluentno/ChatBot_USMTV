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
API_KEY = st.secrets["GEMINI_API_KEY"] 
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
        .inline-logo {
            vertical-align: middle;
            margin: 0 4px;
            transition: transform 0.2s;
            border-radius: 4px;
        }
        .inline-logo:hover {
            transform: scale(1.15); 
        }
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
    timestamp_sekarang = int(time.time())
    url = f"https://usmtv.id/wp-json/wp/v2/posts?per_page=100&_fields=title,link,content,date&_nocache={timestamp_sekarang}"
    
    headers_browser = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    try:
        response = requests.get(url, headers=headers_browser, timeout=30)
        if response.status_code != 200: return False
        data_mentah = response.json()
    except Exception:
        return False
        
    file_path = 'dataset_bersih.csv'
    link_sudah_ada = []
    df_lama = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            df_lama = pd.read_csv(file_path)
            link_sudah_ada = df_lama['link'].tolist()
        except:
            pass

    data_baru = [item for item in data_mentah if item['link'] not in link_sudah_ada]

    if not data_baru:
        return True 
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def bersihkan(teks):
        teks = re.sub(r'<[^>]+>', ' ', str(teks))
        teks = re.sub(r'[^a-zA-Z0-9\s]', '', teks).lower()
        return stemmer.stem(teks)

    data_bersih_baru = []
    for item in data_baru:
        raw = item['content']['rendered']
        data_bersih_baru.append({
            'judul': item['title']['rendered'],
            'link': item['link'],
            'isi_html': raw, 
            'teks_bersih': bersihkan(raw) 
        })

    df_baru = pd.DataFrame(data_bersih_baru)

    if not df_lama.empty:
        df_final = pd.concat([df_baru, df_lama], ignore_index=True)
        df_final = df_final.head(100) 
        df_final.to_csv(file_path, index=False)
    else:
        df_baru.to_csv(file_path, index=False)

    return True

# --- 4. CEK KESEHATAN DATA & LOAD DATA ---
def cek_kesehatan_data():
    file_path = 'dataset_bersih.csv'
    perlu_update = False
    if not os.path.exists(file_path):
        perlu_update = True
    else:
        try:
            umur_file = time.time() - os.path.getmtime(file_path)
            if umur_file > 86400: # Auto update harian
                perlu_update = True
        except:
            perlu_update = True 
    
    if perlu_update:
        # Aku tambahin tulisan loading biar kamu tau dia lagi kerja
        with st.spinner("Mengunduh berita terbaru dari WordPress USMTV. Mohon tunggu..."):
            update_database_otomatis()
            st.cache_resource.clear()

cek_kesehatan_data()

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
    import random 

    if df is None: return "Database bermasalah."

    clean_query = stemmer.stem(pertanyaan_user.lower())
    query_vec = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)

    if similarity_scores.max() < 0.05:
        jumlah_berita = min(3, len(df))
        pertanyaan_kecil = pertanyaan_user.lower()
        if "baru" in pertanyaan_kecil or "hari ini" in pertanyaan_kecil or "today" in pertanyaan_kecil:
            top_indices = list(range(jumlah_berita))
        else:
            top_indices = random.sample(range(len(df)), jumlah_berita)
    else:
        top_indices = similarity_scores.argsort()[0][-3:][::-1]

    konteks_berita = ""
    for idx in top_indices:
        judul = df.iloc[idx]['judul']
        link = df.iloc[idx]['link']
        isi = df.iloc[idx]['isi_html']
        konteks_berita += f"JUDUL: {judul}\nISI: {isi}\nLINK SUMBER: {link}\n\n"

    prompt = f"""
    Kamu adalah USI, asisten USMTV.
    
    KONTEKS BERITA SAAT INI:
    {konteks_berita}
    
    ATURAN MUTLAK:
    1. Jawab pertanyaan HANYA berdasarkan KONTEKS BERITA di atas.
    2. JIKA konteks berita TIDAK NYAMBUNG atau TIDAK ADA kaitannya dengan pertanyaan, KAMU WAJIB MENJAWAB DENGAN RAMAH: "Maaf, dari data berita terbaru USMTV saat ini, saya belum menemukan berita terkait topik tersebut."
    3. JANGAN mengarang jawaban atau memaksakan berita yang tidak relevan.
    4. Di akhir jawaban yang berhasil dan nyambung, WAJIB sertakan: "Sumber: [Link Berita]".
    
    Pertanyaan: {pertanyaan_user}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Maaf, koneksi gangguan."

# --- 6. SIDEBAR PENGATURAN ---
with st.sidebar:
    st.image(ICON_USI, width=80)
    
    st.markdown("<br>", unsafe_allow_html=True) 
    
    st.write("**USI menu options!**")
    
    if st.button("Mau Hapus Chat Aja"):
        st.session_state.messages = []
        st.rerun()

# --- 7. TAMPILAN UTAMA ---
st.title("USI Si Asisten!")

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
        Ingin mengunggah berita anda sendiri? Ayo hubungi Email
        <a href="https://mail.google.com/mail/?view=cm&fs=1&to=usmtv@usm.ac.id" target="_blank" title="Kirim Email via Gmail">
            <img src="data:image/png;base64,{img_email}" height="22" class="inline-logo">
        </a> 
        atau WhatsApp
        <a href="https://wa.me/6287828996924" target="_blank" title="Chat WhatsApp">
            <img src="data:image/png;base64,{img_wa}" height="22" class="inline-logo">
        </a> 
        kami!
    </div>
    """
    st.markdown(info_html, unsafe_allow_html=True)
else:
    st.info("Ingin mengunggah berita anda sendiri? Ayo hubungi Email (usmtv@usm.ac.id) atau WhatsApp (087828996924) kami!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya USI. Ada berita yang dicari?"}]

for message in st.session_state.messages:
    avatar = ICON_USER if message["role"] == "user" else ICON_USI
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya berita..."):
    # 👇 INI DIA CHEAT CODE RAHASIANYA 👇
    if prompt.strip() == "/reset":
        if os.path.exists('dataset_bersih.csv'):
            os.remove('dataset_bersih.csv')
        st.cache_resource.clear()
        st.rerun() # Memaksa bot ngulang dari awal dan nge-download berita baru
    else:
        # LOGIKA NORMAL KALAU BUKAN CHEAT CODE
        st.chat_message("user", avatar=ICON_USER).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner('Mencari...'):
            jawaban_usi = tanya_usi(prompt)
        
        st.chat_message("assistant", avatar=ICON_USI).write(jawaban_usi)
        st.session_state.messages.append({"role": "assistant", "content": jawaban_usi})
