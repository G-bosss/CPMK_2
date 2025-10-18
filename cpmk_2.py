import os
import time
import re
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from webdriver_manager.firefox import GeckoDriverManager
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================
# 1. SETUP FOLDER OUTPUT
# =====================================================
os.makedirs("CPMK 2", exist_ok=True)
SCRAPE_FILE = "CPMK 2/youtube_comments.csv"
CLEAN_FILE = "CPMK 2/cleaned_comments.csv"
NGRAM_FILE = "CPMK 2/bigram_trigram.csv"
WORDCLOUD_FILE = "CPMK 2/wordcloud.png"

# =====================================================
# 2. KONFIGURASI
# =====================================================
VIDEO_URL = "https://www.youtube.com/watch?v=8uy7G2JXVSA"  # Ganti sesuai kebutuhan
MAX_COMMENTS = 500

# =====================================================
# 3. SCRAPING YOUTUBE COMMENTS
# =====================================================
options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

print("🚀 Membuka YouTube...")
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
driver.get(VIDEO_URL)
time.sleep(5)

driver.execute_script("window.scrollTo(0, 600);")
time.sleep(3)

print("🕐 Memuat komentar...")
last_height = driver.execute_script("return document.documentElement.scrollHeight")
scroll_count = 0
start_time = time.time()

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_height == last_height:
        scroll_count += 1
    else:
        scroll_count = 0
        last_height = new_height

    if scroll_count > 5 or (time.time() - start_time > 180):
        break

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

authors = [a.text.strip() for a in soup.select("#author-text span")]
comments = [c.text.strip() for c in soup.select("#content #content-text")]

data = pd.DataFrame({"Author": authors, "Comment": comments})
data = data.head(MAX_COMMENTS)
data.to_csv(SCRAPE_FILE, index=False, encoding="utf-8-sig")
print(f"💾 {len(data)} komentar disimpan ke {SCRAPE_FILE}")

# =====================================================
# 4. PREPROCESSING DAN NORMALISASI
# =====================================================
def clean_text(text):
    text = text.lower()                                # huruf kecil semua
    text = re.sub(r"http\S+|www\S+", "", text)         # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)            # hapus karakter non huruf
    text = re.sub(r"\s+", " ", text).strip()           # hapus spasi berlebih
    return text

def normalize_word(word):
    normalization_dict = {
        "gk": "tidak", "ga": "tidak", "nggak": "tidak",
        "bgt": "banget", "bngt": "banget", "tdk": "tidak",
        "yg": "yang", "aja": "saja", "klo": "kalau", "klu": "kalau",
        "udh": "sudah", "dgn": "dengan", "sm": "sama"
    }
    return normalization_dict.get(word, word)

data["Clean_Comment"] = data["Comment"].apply(clean_text)
data["Clean_Comment"] = data["Clean_Comment"].apply(
    lambda x: " ".join([normalize_word(w) for w in x.split()])
)

# Hapus duplikat
data = data.drop_duplicates(subset=["Clean_Comment"])
data.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
print(f"✅ Komentar bersih disimpan ke {CLEAN_FILE}")

# =====================================================
# 5. BIGRAM DAN TRIGRAM
# =====================================================
vectorizer = CountVectorizer(ngram_range=(2, 3))
X = vectorizer.fit_transform(data["Clean_Comment"])
ngram_counts = X.toarray().sum(axis=0)
ngrams = pd.DataFrame({
    "Ngram": vectorizer.get_feature_names_out(),
    "Count": ngram_counts
}).sort_values(by="Count", ascending=False)

ngrams.to_csv(NGRAM_FILE, index=False, encoding="utf-8-sig")
print(f"📊 Bigram dan Trigram disimpan ke {NGRAM_FILE}")

# =====================================================
# 6. WORDCLOUD
# =====================================================
all_text = " ".join(data["Clean_Comment"])
wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(WORDCLOUD_FILE)
plt.close()
print(f"🌥️ Wordcloud disimpan ke {WORDCLOUD_FILE}")

print("\n🎉 Semua proses selesai! Folder CPMK 2 sudah lengkap.")
