from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
CORS(app)

# ==============================
# MEMORY PERCAKAPAN USER
# ==============================
user_context = {}

followup_words = [
    "harganya",
    "biayanya",
    "berapa duit",
    "berapa duitnya",
    "harga nya",
    "biaya nya"
]

prosedur_words = [
    "prosedur",
    "prosedurnya",
    "bagaimana prosedurnya",
    "cara prosedurnya",
    "langkahnya",
    "gimana prosedurnya",
    "prosedur nya",
    "tahapannya",
    "step nya",
    "caranya gimana",
    "gimana caranya"
]

# KATA NEGASI - UNTUK MENDETEKSI PENYANGKALAN (TIDAK, BUKAN, DLL)
negation_words = [
    "tidak", "tdk", "tdak", "bukan", "enggak", "engga", "gak", "ga",
    "nggak", "ngga", "kagak", "kaga", "blm", "belum", "jangan",
    "tanpa", "bkn", "tak", "ngak", "x sensitif", "nggk", "udah engga",
    "sudah tidak", "bukannya"
]

# KEYWORD PEMBUKA UMUM - KETIKA USER BELUM SPESIFIK SEBUTKAN LAYANAN
pembuka_umum_keywords = [
    "saya mau perawatan", "saya ingin perawatan", "mau perawatan gigi",
    "ingin perawatan gigi", "saya mau ke klinik", "saya ingin ke klinik",
    "mau periksa gigi", "ingin periksa gigi", "mau cek gigi",
    "ingin konsultasi gigi", "saya butuh perawatan", "butuh perawatan gigi",
    "mau rawat gigi", "ingin rawat gigi", "saya ada masalah gigi",
    "gigi saya bermasalah", "mau berobat gigi", "ingin berobat gigi",
    "mau ke dokter gigi", "ingin ke dokter gigi", "mau periksa ke klinik",
    "saya pengen rawat gigi", "pengen rawat gigi", "pengen periksa gigi",
    "pengen ke klinik", "mau dirawat", "ingin dirawat", "perlu perawatan gigi",
    "mau perawatan", "ingin perawatan", "mau treatment gigi",
    "saya ada keluhan gigi", "ada keluhan dengan gigi saya",
    "gigi saya kenapa kenapa", "tolong bantu gigi saya",
    "saya butuh bantuan untuk gigi", "mau konsul gigi", "ingin konsul gigi",
    "mau nanya soal gigi", "mau tanya soal gigi", "boleh tanya soal gigi",
    "saya mau tanya tentang gigi", "saya ada problem gigi",
    "gigi saya ada masalah", "mau servis gigi", "pengen benerin gigi",
    "mau benerin gigi", "ingin benerin gigi"
]

# KEYWORD PERTANYAAN TERBUKA / TIDAK JELAS ARAHNYA
pertanyaan_terbuka_keywords = [
    "jadi saya harus gimana", "saya harus gimana", "terus gimana",
    "lalu gimana", "jadi bagaimana", "saya harus bagaimana",
    "terus bagaimana", "lalu bagaimana", "jadi apa yang harus saya lakukan",
    "apa yang harus saya lakukan", "saya harus apa", "terus saya harus apa",
    "lalu saya harus apa", "jadi harus gimana", "trus gimana",
    "trus harus gimana", "kalau begitu gimana", "kalo begitu gimana",
    "lantas gimana", "lantas bagaimana", "next nya gimana",
    "selanjutnya gimana", "habis ini gimana", "setelah ini gimana",
    "terus apa lagi", "lalu apa lagi", "jadi apa", "terus apa",
    "saya bingung", "saya tidak tahu harus apa", "gatau harus gimana",
    "ga tau harus gimana", "bagusnya gimana", "baiknya gimana",
    "sarannya apa", "menurut kamu gimana", "menurut bot gimana"
]

# KEYWORD PERTANYAAN SEPUTAR DOKTER (UNTUK KLARIFIKASI PROFESI)
dokter_keywords = [
    "dokter gigi", "dokter", "ada dokter", "siapa dokternya",
    "dokternya siapa", "yang praktik dokter", "ditangani dokter",
    "dokter giginya siapa", "apakah ada dokter", "ada dokternya tidak",
    "dokter atau perawat", "yang menangani siapa", "siapa yang menangani"
]

# JAWABAN UNTUK PERTANYAAN TERBUKA
jawaban_pertanyaan_terbuka = "Untuk membantu Anda lebih tepat, boleh sebutkan layanan yang Anda butuhkan? Kami menyediakan scaling, tambal gigi, cabut gigi, bleaching, premedikasi, dan konsultasi 😊"

# KEYWORD BEHEL - SOAL HARGA/BIAYA (DICEK LEBIH DULU SEBELUM KETERSEDIAAN UMUM)
behel_harga_keywords = [
    "berapa harga behel", "berapa biaya behel", "harga behel",
    "biaya behel", "behel berapa", "behel mahal", "tarif behel",
    "behel berapaan", "behel berapa duit"
]

# KEYWORD BEHEL - SOAL KETERSEDIAAN/UMUM
behel_keywords = ["behel", "kawat gigi"]

# KEYWORD GIGI KEROPOS (JAWABAN KHUSUS, BUKAN TEMPLATE TAMBAL UMUM)
keropos_keywords = ["keropos"]

# KEYWORD PLAK GIGI (JAWABAN KHUSUS, BUKAN TEMPLATE SCALING UMUM)
plak_keywords = ["plak"]

# KEYWORD GIGI GOYANG (JAWABAN KHUSUS, TIDAK MENYIMPULKAN HARUS CABUT)
goyang_keywords = ["goyang", "longgar"]

# ==============================
# PREPROCESSING TEXT
# ==============================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    replacements = {
        "bersihin": "scaling",
        "karang": "scaling",
        "bersih karang": "scaling",
        "daftar": "booking",
        "reservasi": "booking",
        "antri": "booking"
    }

    for key, val in replacements.items():
        text = text.replace(key, val)

    return text

# ==============================
# DATASET PERTANYAAN & JAWABAN
# ==============================
qa_pairs = [
    {"q": "halo", "a": "Halo 😊 Selamat datang di layanan informasi Rizki Dental."},
    {"q": "hai", "a": "Hai 😊 Ada yang bisa kami bantu?"},
    {"q": "selamat malam", "a": "Halo 😊 Silakan tanyakan informasi layanan klinik."},
    {"q": "permisi", "a": "Halo 😊 Kami siap membantu Anda."},

    {"q": "jam buka klinik", "a": "Klinik buka setiap Senin sampai Sabtu pukul 16.00–21.00 WIB 😊"},
    {"q": "klinik buka jam berapa", "a": "Jam operasional Rizki Dental adalah pukul 16.00–21.00 WIB."},
    {"q": "jadwal klinik", "a": "Kami melayani pasien mulai pukul 16.00–21.00 WIB."},

    {"q": "hari apa klinik buka", "a": "Klinik buka dari hari Senin sampai Sabtu 😊"},
    {"q": "klinik buka hari apa", "a": "Rizki Dental melayani pasien setiap Senin–Sabtu."},

    {"q": "alamat klinik dimana", "a": "Alamat Rizki Dental berada di Jl. Gajah No 57 Pulo Ara Geudong Teungoh Kota Juang."},
    {"q": "lokasi klinik", "a": "Lokasi klinik berada di Jl. Gajah No 57 Pulo Ara Geudong Teungoh Kota Juang."},

    {"q": "kontak klinik", "a": "Silakan hubungi admin klinik melalui WhatsApp untuk informasi lebih lanjut 😊"},
    {"q": "nomor whatsapp", "a": "Silakan hubungi admin klinik melalui WhatsApp."},
    {"q": "nomor klinik", "a": "Kontak klinik dapat dihubungi melalui WhatsApp admin."},

    {"q": "layanan apa saja", "a": "Layanan tersedia meliputi scaling, tambal gigi, cabut gigi, bleaching, premedikasi, dan konsultasi 😊"},
    {"q": "apa saja layanan di klinik", "a": "Rizki Dental menyediakan layanan scaling, tambal gigi, cabut gigi, bleaching, premedikasi, dan konsultasi."},

    {"q": "cara merawat gigi", "a": "Sikat gigi minimal 2 kali sehari dan lakukan kontrol rutin setiap 6 bulan 😊"},
    {"q": "tips merawat gigi", "a": "Gunakan pasta gigi berfluoride dan kurangi makanan manis agar gigi tetap sehat."},
    {"q": "bagaimana cara menjaga kesehatan gigi", "a": "Jaga kesehatan gigi dengan sikat gigi rutin, kurangi minuman bersoda, dan kontrol ke klinik setiap 6 bulan 😊"},

    {"q": "kenapa gigi berlubang", "a": "Gigi berlubang umumnya disebabkan oleh bakteri dan penumpukan plak pada permukaan gigi 😊"},
    {"q": "penyebab gigi berlubang", "a": "Penyebab utama gigi berlubang adalah bakteri, sisa makanan manis yang menumpuk, dan kurangnya kebersihan gigi."},
    {"q": "apa penyebab gigi berlubang", "a": "Gigi berlubang terjadi karena plak bakteri menghasilkan asam yang mengikis email gigi secara perlahan."},
    {"q": "mengapa gigi bisa berlubang", "a": "Konsumsi makanan manis dan kurang menjaga kebersihan gigi dapat mempercepat terjadinya gigi berlubang 😊"},

    {"q": "waktu ideal scaling gigi", "a": "Waktu ideal scaling gigi adalah setiap 6 bulan sekali untuk mencegah penumpukan karang gigi dan menjaga kesehatan gusi 😊"},
    {"q": "kapan harus scaling gigi", "a": "Disarankan melakukan scaling gigi minimal 6 bulan sekali agar karang gigi tidak menumpuk."},
    {"q": "berapa kali scaling dalam setahun", "a": "Scaling sebaiknya dilakukan 1–2 kali dalam setahun tergantung kondisi kebersihan gigi masing-masing pasien 😊"},
    {"q": "seberapa sering scaling gigi", "a": "Idealnya scaling dilakukan setiap 6 bulan sekali untuk mencegah penumpukan karang gigi 😊"},
    {"q": "kapan waktu scaling gigi", "a": "Waktu ideal scaling gigi adalah setiap 6 bulan sekali 😊"},
    {"q": "kapan scaling gigi", "a": "Disarankan scaling setiap 6 bulan sekali agar karang gigi tidak menumpuk 😊"},
    {"q": "jadwal scaling gigi", "a": "Jadwal scaling yang ideal adalah setiap 6 bulan sekali 😊"},
    {"q": "frekuensi scaling gigi", "a": "Frekuensi scaling yang disarankan adalah 1–2 kali dalam setahun 😊"},

    {"q": "tips memilih sikat gigi", "a": "Pilih sikat gigi dengan bulu halus (soft bristle) dan kepala sikat yang kecil agar dapat menjangkau seluruh area gigi. Ganti sikat gigi setiap 3 bulan sekali 😊"},
    {"q": "cara memilih sikat gigi yang benar", "a": "Sikat gigi yang baik adalah yang berbulu lembut, berukuran sesuai mulut, dan nyaman digenggam. Jangan lupa ganti setiap 3 bulan 😊"},
    {"q": "sikat gigi yang bagus seperti apa", "a": "Gunakan sikat gigi berbulu halus untuk menghindari iritasi gusi. Pilih ukuran kepala sikat yang kecil agar lebih mudah membersihkan gigi belakang 😊"},
    {"q": "sikat gigi yang baik", "a": "Sikat gigi yang baik memiliki bulu lembut dan kepala kecil. Ganti sikat gigi setiap 3 bulan sekali atau saat bulu sikat sudah mekar 😊"},

    {"q": "makanan yang baik untuk gigi", "a": "Makanan yang baik untuk kesehatan gigi antara lain susu, keju, sayuran hijau, dan buah-buahan berserat seperti apel yang membantu membersihkan gigi secara alami 😊"},
    {"q": "makanan sehat untuk gigi", "a": "Konsumsi makanan kaya kalsium seperti susu dan keju untuk memperkuat email gigi. Hindari makanan dan minuman manis berlebihan 😊"},
    {"q": "makanan apa yang bagus untuk gigi", "a": "Makanan sehat untuk gigi meliputi produk susu, ikan, kacang-kacangan, dan sayuran hijau yang kaya kalsium dan fosfor 😊"},
    {"q": "nutrisi untuk kesehatan gigi", "a": "Buah-buahan berserat seperti apel dan wortel baik untuk gigi karena membantu membersihkan plak secara alami 😊"},

    {"q": "cara mengatasi gigi sensitif", "a": "Gigi sensitif dapat diatasi dengan menggunakan pasta gigi khusus sensitif, menghindari makanan/minuman terlalu panas atau dingin"},
    {"q": "gigi sensitif", "a": "Untuk mengatasi gigi sensitif, gunakan pasta gigi sensitif dan sikat gigi berbulu lembut. Hindari minuman asam dan terlalu dingin 😊"},
    {"q": "gigi ngilu", "a": "Gigi ngilu biasanya disebabkan oleh email gigi yang terkikis atau gusi yang turun. Gunakan pasta gigi sensitif dan segera konsultasikan ke klinik 😊"},
    {"q": "kenapa gigi ngilu", "a": "Gigi sensitif atau ngilu dapat disebabkan oleh karang gigi, email tipis, atau gusi bermasalah. Disarankan untuk melakukan pemeriksaan langsung ke klinik 😊"},

    {"q": "booking", "a": "Reservasi dapat dilakukan melalui WhatsApp admin klinik 😊"},
    {"q": "cara daftar", "a": "Silakan hubungi admin klinik untuk melakukan pendaftaran melalui WhatsApp 😊"},
    {"q": "cara reservasi", "a": "Pendaftaran dilakukan melalui WhatsApp klinik 😊"},
    {"q": "cara antri", "a": "Silakan melakukan booking melalui WhatsApp admin klinik."},

    {"q": "premedikasi", "a": "Premedikasi adalah pemberian obat sebelum tindakan gigi untuk mengurangi rasa sakit atau risiko infeksi. Jenis obat yang diberikan disesuaikan dengan kondisi pasien berdasarkan pertimbangan tenaga kesehatan yang menangani. Untuk penanganan lebih lanjut, pasien disarankan melakukan konsultasi langsung ke klinik 😊"},
    {"q": "apa itu premedikasi", "a": "Premedikasi adalah pemberian obat sebelum tindakan gigi untuk mengurangi rasa sakit atau risiko infeksi. Jenis obat yang diberikan disesuaikan dengan kondisi pasien berdasarkan pertimbangan tenaga kesehatan yang menangani. Untuk penanganan lebih lanjut, pasien disarankan melakukan konsultasi langsung ke klinik 😊"},
    {"q": "layanan premedikasi", "a": "Premedikasi adalah pemberian obat sebelum tindakan gigi untuk mengurangi rasa sakit atau risiko infeksi. Jenis obat yang diberikan disesuaikan dengan kondisi pasien berdasarkan pertimbangan tenaga kesehatan yang menangani. Untuk penanganan lebih lanjut, pasien disarankan melakukan konsultasi langsung ke klinik 😊"},

    {"q": "konsultasi", "a": "Rizki Dental menyediakan layanan konsultasi gigi. Silakan hubungi admin melalui WhatsApp untuk membuat jadwal konsultasi 😊"},
    {"q": "layanan konsultasi", "a": "Rizki Dental menyediakan layanan konsultasi gigi. Silakan hubungi admin melalui WhatsApp untuk membuat jadwal konsultasi 😊"},
    {"q": "cara konsultasi", "a": "Untuk konsultasi gigi, silakan hubungi admin Rizki Dental melalui WhatsApp untuk membuat jadwal 😊"},
]

questions = [p["q"] for p in qa_pairs]
answers = [p["a"] for p in qa_pairs]

# ==============================
# FIT VECTORIZER SEKALI SAAT STARTUP
# ==============================
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ==============================
# API CHATBOT
# ==============================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    user_message_raw = data['message']
    user_message = preprocess(user_message_raw)

    # AMBIL user_id DARI REQUEST
    user_id = data.get('user_id', 'default_user')

    # ==============================
    # KLARIFIKASI PROFESI (DOKTER VS PERAWAT GIGI)
    # ==============================
    if any(word in user_message for word in dokter_keywords):
        return jsonify({
            "reply": "Di Rizki Dental, layanan ditangani langsung oleh perawat gigi yang berpengalaman, bukan dokter gigi 😊"
        })

    # ==============================
    # DETEKSI PERTANYAAN SEPUTAR BEHEL
    # ==============================
    if any(word in user_message for word in behel_harga_keywords):
        return jsonify({
            "reply": "Untuk estimasi biaya behel, silakan tanyakan langsung ke admin melalui WhatsApp ya, karena biaya bisa berbeda sesuai jenis behel yang dipilih 😊"
        })

    if any(word in user_message for word in behel_keywords):
        return jsonify({
            "reply": "Layanan behel tersedia di Rizki Dental. Untuk informasi lebih lanjut, silakan hubungi admin melalui WhatsApp 😊"
        })

    # ==============================
    # DETEKSI GIGI KEROPOS (JAWABAN KHUSUS)
    # ==============================
    if any(word in user_message for word in keropos_keywords):
        return jsonify({
            "reply": "Gigi keropos biasanya disebabkan oleh kerusakan email gigi akibat bakteri atau kekurangan mineral. Disarankan untuk segera memeriksakan ke klinik agar dapat ditangani dengan tambal gigi atau perawatan lain yang sesuai 😊"
        })

    # ==============================
    # DETEKSI PLAK GIGI (JAWABAN KHUSUS)
    # ==============================
    if any(word in user_message for word in plak_keywords):
        return jsonify({
            "reply": "Plak yang menumpuk di gigi dapat dibersihkan melalui prosedur scaling, yaitu pembersihan karang gigi untuk menjaga kesehatan gusi dan kebersihan mulut 😊"
        })

    # ==============================
    # DETEKSI GIGI GOYANG (JAWABAN UMUM, TIDAK MENYIMPULKAN HARUS CABUT)
    # ==============================
    if any(word in user_message for word in goyang_keywords):
        return jsonify({
            "reply": "Gigi goyang dapat disebabkan oleh berbagai faktor seperti gigi susu yang akan tanggal, cedera, atau masalah pada jaringan gigi. Disarankan untuk memeriksakan langsung ke klinik agar dapat ditentukan penanganan yang tepat 😊"
        })

    # ==============================
    # DETEKSI PERTANYAAN TERBUKA / TIDAK JELAS ARAHNYA
    # ==============================
    if any(word in user_message for word in pertanyaan_terbuka_keywords):
        return jsonify({
            "reply": jawaban_pertanyaan_terbuka
        })

    # ==============================
    # DETEKSI PEMBUKA UMUM (BELUM SPESIFIK LAYANAN)
    # ==============================
    if any(word in user_message for word in pembuka_umum_keywords):
        return jsonify({
            "reply": "Tentu, kami siap membantu 😊 Perawatan apa yang Anda butuhkan? Kami menyediakan layanan scaling, tambal gigi, cabut gigi, bleaching, premedikasi, dan konsultasi."
        })

    # ==============================
    # DETEKSI PERTANYAAN DIAGNOSIS
    # ==============================
    diagnosis_keywords = [
        "diagnosa", "diagnosis", "obat apa",
        "gigi saya sakit", "kenapa gigi saya",
        "apakah saya", "penyakit gigi"
    ]

    if any(word in user_message for word in diagnosis_keywords):
        return jsonify({
            "reply": "Maaf 😊 Chatbot ini hanya menyediakan informasi layanan klinik dan tidak dapat memberikan diagnosis medis. Silakan hubungi admin atau datang langsung ke klinik untuk pemeriksaan lebih lanjut."
        })

    # ==============================
    # DETEKSI PROSEDUR LANGSUNG
    # ==============================
    prosedur_langsung_scaling = [
        "prosedur scaling", "bagaimana prosedur scaling",
        "cara scaling", "proses scaling"
    ]
    prosedur_langsung_tambal = [
        "prosedur tambal", "bagaimana prosedur tambal",
        "cara tambal gigi", "proses tambal gigi"
    ]
    prosedur_langsung_cabut = [
        "prosedur cabut", "bagaimana prosedur cabut",
        "cara cabut gigi", "proses cabut gigi"
    ]
    prosedur_langsung_bleaching = [
        "prosedur bleaching", "bagaimana prosedur bleaching",
        "cara bleaching", "proses bleaching"
    ]

    if any(word in user_message for word in prosedur_langsung_scaling):
        user_context[user_id] = "scaling"
        return jsonify({"reply": "Prosedur scaling dilakukan dengan membersihkan karang gigi menggunakan alat khusus. Pasien tidak perlu khawatir karena prosedur ini umumnya tidak sakit dan berlangsung sekitar 30–60 menit 😊"})

    if any(word in user_message for word in prosedur_langsung_tambal):
        user_context[user_id] = "tambal"
        return jsonify({"reply": "Prosedur tambal gigi dilakukan dengan membersihkan bagian gigi yang berlubang, kemudian mengisi lubang dengan bahan tambal. Dapat dilakukan dengan anestesi sesuai kondisi pasien 😊"})

    if any(word in user_message for word in prosedur_langsung_cabut):
        user_context[user_id] = "cabut"
        return jsonify({"reply": "Prosedur cabut gigi diawali dengan pemeriksaan kondisi pasien. Selanjutnya dilakukan pemberian anestesi lokal sesuai kebutuhan agar pasien merasa lebih nyaman selama tindakan. Setelah itu, proses pencabutan gigi dilakukan sesuai prosedur pelayanan yang berlaku di klinik 😊"})

    if any(word in user_message for word in prosedur_langsung_bleaching):
        user_context[user_id] = "bleaching"
        return jsonify({"reply": "Prosedur bleaching dilakukan dengan mengaplikasikan bahan pemutih gigi pada permukaan gigi. Prosedur ini berlangsung sekitar 60–90 menit dan hasil pemutihan dapat bertahan beberapa bulan 😊"})

    # ==============================
    # DETEKSI BIAYA LANGSUNG
    # ==============================
    biaya_scaling_keywords = [
        "berapa biaya scaling", "harga scaling", "biaya scaling",
        "berapa harga scaling", "tarif scaling", "biaya scaling gigi"
    ]
    biaya_tambal_keywords = [
        "berapa biaya tambal", "harga tambal", "biaya tambal",
        "berapa harga tambal", "tarif tambal", "biaya tambal gigi"
    ]
    biaya_cabut_keywords = [
        "berapa biaya cabut", "harga cabut", "biaya cabut",
        "berapa harga cabut", "tarif cabut", "biaya cabut gigi"
    ]
    biaya_bleaching_keywords = [
        "berapa biaya bleaching", "harga bleaching", "biaya bleaching",
        "berapa harga bleaching", "tarif bleaching"
    ]
   
 
    if any(word in user_message for word in biaya_scaling_keywords):
        user_context[user_id] = "scaling"
        return jsonify({"reply": "Estimasi biaya scaling di Rizki Dental mulai dari Rp200.000 😊"})

    if any(word in user_message for word in biaya_tambal_keywords):
        user_context[user_id] = "tambal"
        return jsonify({"reply": "Estimasi biaya tambal gigi mulai dari Rp200.000 😊"})

    if any(word in user_message for word in biaya_cabut_keywords):
        user_context[user_id] = "cabut"
        return jsonify({"reply": "Estimasi biaya cabut gigi mulai dari Rp200.000 😊"})

    if any(word in user_message for word in biaya_bleaching_keywords):
        user_context[user_id] = "bleaching"
        return jsonify({"reply": "Estimasi biaya bleaching mulai dari Rp1.500.000 😊"})

    

    

    # ==============================
    # KEYWORD EDUKASI
    # ==============================
    sensitif_penyebab_keywords = [
        "kenapa gigi ngilu", "gigi ngilu kenapa", "mengapa gigi ngilu",
        "kenapa gigi sensitif", "gigi sensitif kenapa", "mengapa gigi sensitif",
        "kenapa gigi nyeri", "gigi nyeri kenapa", "kenapa gigi linu", "gigi linu kenapa"
    ]
    sensitif_keywords = [
        "gigi sensitif", "gigi ngilu", "mengatasi gigi sensitif",
        "cara mengatasi gigi sensitif",
        "gigi terasa ngilu", "gigi terasa sensitif",
        "bagaimana cara mengatasi gigi sensitif",
        "gigi nyeri", "gigi linu", "gigi ngilu kalau minum dingin",
        "gigi sakit kalau minum dingin", "gigi ngilu makan dingin",
        "gigi sensitif terhadap dingin", "gigi sensitif terhadap panas"
    ]
    penyebab_keywords = [
        "penyebab gigi berlubang", "kenapa gigi berlubang",
        "mengapa gigi berlubang", "apa penyebab gigi berlubang",
        "gigi berlubang kenapa", "gigi berlubang penyebabnya"
    ]
    waktu_scaling_keywords = [
        "waktu ideal scaling", "kapan harus scaling",
        "kapan waktu scaling", "kapan scaling",
        "seberapa sering scaling", "berapa kali scaling",
        "jadwal scaling", "frekuensi scaling"
    ]
    sikat_gigi_keywords = [
        "tips memilih sikat gigi", "cara memilih sikat gigi",
        "sikat gigi yang bagus", "sikat gigi yang baik",
        "memilih sikat gigi"
    ]
    makanan_keywords = [
        "makanan yang baik untuk gigi", "makanan sehat untuk gigi",
        "makanan bagus untuk gigi", "nutrisi untuk gigi",
        "makanan untuk kesehatan gigi"
    ]

    # Cek apakah user menyangkal punya gigi sensitif (mengandung kata negasi + sensitif/ngilu)
    has_negation = any(re.search(r'\b' + re.escape(neg) + r'\b', user_message) for neg in negation_words)
    mentions_sensitif_topic = ("sensitif" in user_message) or ("ngilu" in user_message) or ("linu" in user_message) or ("nyeri" in user_message)

    # Cek dulu apakah user nanya PENYEBAB (kenapa), bukan cara mengatasi
    if any(word in user_message for word in sensitif_penyebab_keywords):
        return jsonify({"reply": "Gigi sensitif atau ngilu dapat disebabkan oleh karang gigi, email gigi yang terkikis, atau gusi yang turun. Disarankan untuk melakukan pemeriksaan langsung ke klinik 😊"})

    if any(word in user_message for word in sensitif_keywords) and not (has_negation and mentions_sensitif_topic):
        return jsonify({"reply": "Gigi sensitif dapat diatasi dengan menggunakan pasta gigi khusus sensitif, menghindari makanan/minuman terlalu panas atau dingin"})

    if has_negation and mentions_sensitif_topic:
        return jsonify({"reply": "Baik, kalau begitu ada keluhan atau layanan lain yang ingin Anda tanyakan? Kami menyediakan scaling, tambal gigi, cabut gigi, bleaching, premedikasi, dan konsultasi 😊"})

    if any(word in user_message for word in penyebab_keywords):
        return jsonify({"reply": "Penyebab utama gigi berlubang adalah bakteri dan penumpukan plak akibat sisa makanan manis yang tidak dibersihkan. Kurangnya menjaga kebersihan gigi juga mempercepat proses ini 😊"})

    if any(word in user_message for word in waktu_scaling_keywords):
        return jsonify({"reply": "Waktu ideal scaling gigi adalah setiap 6 bulan sekali untuk mencegah penumpukan karang gigi dan menjaga kesehatan gusi 😊"})

    if any(word in user_message for word in sikat_gigi_keywords):
        return jsonify({"reply": "Pilih sikat gigi dengan bulu halus (soft bristle) dan kepala sikat yang kecil agar dapat menjangkau seluruh area gigi. Ganti sikat gigi setiap 3 bulan sekali 😊"})

    if any(word in user_message for word in makanan_keywords):
        return jsonify({"reply": "Makanan yang baik untuk kesehatan gigi antara lain susu, keju, sayuran hijau, dan buah berserat seperti apel yang membantu membersihkan gigi secara alami 😊"})

    # ==============================
    # KEYWORD LAYANAN
    # ==============================
    scaling_keywords = [
        "scaling", "scalling", "bersihin karang", "bersih karang",
        "bersihkan karang", "karang gigi", "gigi kotor", "banyak karang",
        "membersihkan gigi", "pembersihan gigi", "penuh karang",
        "ada karangnya", "kuning karang", "bersihin karang"
    ]
    tambal_keywords = [
        "tambal", "nambal", "bolong", "berlubang", "lubang di gigi",
        "rapuh", "patah sebagian", "retak", "bolong di gigi"
    ]
    cabut_keywords = [
        "cabut", "mau dicabut", "pencabutan", "harus dicabut",
        "mau lepas", "copot", "cabutkan gigi", "gigi bungsu",
        "geraham sakit parah"
    ]
    bleaching_keywords = [
        "bleaching", "memutihkan", "kuning", "kusam", "pemutihan",
        "gigi putih", "kurang putih", "tidak putih", "gigi cerah",
        "whitening"
    ]
    premedikasi_keywords = ["premedikasi", "obat sebelum tindakan", "obat sebelum cabut"]
    konsultasi_keywords = ["konsultasi", "mau konsultasi", "ingin konsultasi", "tanya dokter", "tanya ke klinik"]

    # ==============================
    # FOLLOW UP — BIAYA & PROSEDUR
    # ==============================
    is_followup = any(word in user_message for word in followup_words)
    is_prosedur = any(word in user_message for word in prosedur_words)

    if user_id in user_context:
        last_topic = user_context[user_id]

        if is_followup:
            if last_topic == "scaling":
                return jsonify({"reply": "Estimasi biaya scaling di Rizki Dental mulai dari Rp200.000 😊"})
            elif last_topic == "tambal":
                return jsonify({"reply": "Estimasi biaya tambal gigi mulai dari Rp200.000 😊"})
            elif last_topic == "cabut":
                return jsonify({"reply": "Estimasi biaya cabut gigi mulai dari Rp200.000 😊"})
            elif last_topic == "bleaching":
                return jsonify({"reply": "Estimasi biaya bleaching mulai dari Rp1.500.000 😊"})

        if is_prosedur:
            if last_topic == "scaling":
                return jsonify({"reply": "Prosedur scaling dilakukan dengan membersihkan karang gigi menggunakan alat khusus. Pasien tidak perlu khawatir karena prosedur ini umumnya tidak sakit dan berlangsung sekitar 30–60 menit 😊"})
            elif last_topic == "tambal":
                return jsonify({"reply": "Prosedur tambal gigi dilakukan dengan membersihkan bagian gigi yang berlubang, kemudian mengisi lubang dengan bahan tambal. Dapat dilakukan dengan anestesi sesuai kondisi pasien 😊"})
            elif last_topic == "cabut":
                return jsonify({"reply": "Prosedur cabut gigi diawali dengan pemeriksaan kondisi pasien. Selanjutnya dilakukan pemberian anestesi lokal sesuai kebutuhan agar pasien merasa lebih nyaman selama tindakan. Setelah itu, proses pencabutan gigi dilakukan sesuai prosedur pelayanan yang berlaku di klinik 😊"})
            elif last_topic == "bleaching":
                return jsonify({"reply": "Prosedur bleaching dilakukan dengan mengaplikasikan bahan pemutih gigi pada permukaan gigi. Prosedur ini berlangsung sekitar 60–90 menit dan hasil pemutihan dapat bertahan beberapa bulan 😊"})

    # ==============================
    # INTENT DETECTION
    # ==============================
    if any(word in user_message for word in scaling_keywords):
        user_context[user_id] = "scaling"
        if "sakit" in user_message:
            return jsonify({"reply": "Scaling umumnya tidak sakit 😊 Namun beberapa pasien mungkin merasakan sedikit tidak nyaman saat proses pembersihan karang gigi."})
        return jsonify({"reply": "Scaling adalah prosedur pembersihan karang gigi untuk menjaga kesehatan gusi dan kebersihan mulut 😊"})

    if any(word in user_message for word in tambal_keywords):
        user_context[user_id] = "tambal"
        if "sakit" in user_message:
            return jsonify({"reply": "Tambal gigi umumnya tidak sakit karena dapat dilakukan dengan anestesi sesuai kondisi pasien 😊"})
        return jsonify({"reply": "Tambal gigi dilakukan untuk memperbaiki gigi yang berlubang atau rusak 😊"})

    if any(word in user_message for word in cabut_keywords):
        user_context[user_id] = "cabut"
        if "sakit" in user_message:
            return jsonify({"reply": "Cabut gigi biasanya dilakukan dengan anestesi sehingga pasien lebih nyaman 😊"})
        return jsonify({"reply": "Cabut gigi dilakukan apabila kondisi gigi sudah tidak dapat dipertahankan 😊"})

    if any(word in user_message for word in bleaching_keywords):
        user_context[user_id] = "bleaching"
        return jsonify({"reply": "Bleaching adalah prosedur pemutihan gigi untuk membantu membuat warna gigi tampak lebih cerah 😊"})

    if any(word in user_message for word in premedikasi_keywords):
        return jsonify({"reply": "Premedikasi adalah pemberian obat sebelum tindakan gigi untuk mengurangi rasa sakit atau risiko infeksi. Jenis obat yang diberikan disesuaikan dengan kondisi pasien berdasarkan pertimbangan tenaga kesehatan yang menangani. Untuk penanganan lebih lanjut, pasien disarankan melakukan konsultasi langsung ke klinik 😊"})

    if any(word in user_message for word in konsultasi_keywords):
        return jsonify({"reply": "Rizki Dental menyediakan layanan konsultasi gigi. Silakan hubungi admin melalui WhatsApp untuk membuat jadwal konsultasi 😊"})

    # ==============================
    # TF-IDF & COSINE SIMILARITY
    # ==============================
    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    score = similarity.max()
    index = similarity.argmax()

    if score < 0.15:
        response = (
            "Maaf 😊 Saya belum memahami pertanyaan Anda. "
            "Silakan tanyakan seputar layanan, jam operasional, "
            "reservasi, atau informasi klinik."
        )
    else:
        response = answers[index]

    return jsonify({"reply": response})

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
