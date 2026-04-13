import streamlit as st
import pandas as pd
import requests
import re
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

# --- 3. PIPELINE DATA OTOMATIS (Tanpa CSV, Super Cepat) ---
# TTL = 3600 detik (1 Jam). Data otomatis refresh di RAM server setiap 1 jam!
@st.cache_resource(ttl=3600, show_spinner="USI sedang sinkronisasi berita terbaru USMTV hari ini...")
def siapkan_otak_usi():
    timestamp_sekarang = int(time.time())
    # Kita tarik 50 berita aja biar proses pre-processing Sastrawi lebih ngebut (gak lemot)
    url = f"https://usmtv.id/wp-json/wp/v2/posts?per_page=50&_fields=title,link,content&_nocache={timestamp_sekarang}"
    
    headers_browser = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers_browser, timeout=30)
        if response.status_code != 200: return None, None, None, None
        data_mentah = response.json()
    except Exception:
        return None, None, None, None

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

    df = pd.DataFrame(data_bersih)
    df['teks_bersih'] = df['teks_bersih'].fillna('') 

    vectorizer = TfidfVectorizer(ngram_range=(1, 2)) 
    tfidf_matrix = vectorizer.fit_transform(df['teks_bersih'])
    
    return df, vectorizer, tfidf_matrix, stemmer

# Jalankan pipeline
df, vectorizer, tfidf_matrix, stemmer = siapkan_otak_usi()

# --- 4. LOGIKA USI ---
def tanya_usi(pertanyaan_user):
    import random 

    if df is None or df.empty: 
        return "Maaf, koneksi ke database USMTV sedang gangguan. Coba beberapa saat lagi."

    clean_query = stemmer.stem(pertanyaan_user.lower())
    query_vec = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)

    if similarity_scores.max() < 0.05:
        jumlah_berita = min(3, len(df))
        pertanyaan_kecil = pertanyaan_user.lower()
        # Kalau user minta berita baru, kasih 3 indeks teratas (terbaru dari WP)
        if "baru" in pertanyaan_kecil or "hari ini" in pertanyaan_kecil or "today" in pertanyaan_kecil or "update" in pertanyaan_kecil:
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
    Kamu adalah USI, asisten pintar dari portal berita USMTV.
    
    KONTEKS BERITA SAAT INI (Update Hari Ini):
    {konteks_berita}
    
    ATURAN MUTLAK:
    1. Jawab pertanyaan HANYA menggunakan informasi dari KONTEKS BERITA di atas.
    2. JIKA konteks berita tidak relevan dengan pertanyaan, JANGAN NGARANG. Jawab dengan ramah: "Maaf, dari data berita terbaru USMTV hari ini, saya belum menemukan berita terkait topik tersebut."
    3. Jika menjawab berdasarkan konteks, WAJIB sertakan "Sumber: [Link Berita]" di akhir.
    
    Pertanyaan: {pertanyaan_user}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Maaf, koneksi ke otak AI sedang gangguan."

# --- 5. SIDEBAR PENGATURAN ---
with st.sidebar:
    st.image(ICON_USI, width=80)
    st.markdown("<br>", unsafe_allow_html=True) 
    st.write("**USI menu options!**")
    
    if st.button("Mau Hapus Chat Aja"):
        st.session_state.messages = []
        st.rerun()

# --- 6. TAMPILAN UTAMA ---
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
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya USI. Ada berita yang dicari hari ini?"}]

for message in st.session_state.messages:
    avatar = ICON_USER if message["role"] == "user" else ICON_USI
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanya berita hari ini..."):
    st.chat_message("user", avatar=ICON_USER).write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner('Mencari di tumpukan berita USMTV...'):
        jawaban_usi = tanya_usi(prompt)
    
    st.chat_message("assistant", avatar=ICON_USI).write(jawaban_usi)
    st.session_state.messages.append({"role": "assistant", "content": jawaban_usi})
