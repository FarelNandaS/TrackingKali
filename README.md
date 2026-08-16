# 🖐️ TrackingKali

Aplikasi *Computer Vision* sederhana berbasis Python untuk mendeteksi dan melacak gerakan jari tangan secara *real-time*.

### ✨ Fitur Utama:
* **Real-time Tracking:** Melacak 21 titik *landmark* tangan secara presisi menggunakan **MediaPipe Tasks API**.
* **Gesture Identification:** Mengidentifikasi bentuk gestur jari (seperti *Peace Sign*, *Fist/Mengepal*, dll).
* **Dynamic Image Overlay:** Menampilkan gambar atau stiker khusus di layar yang mengikuti posisi jari secara interaktif.

## ⚙ Requierments
- Python 3.13.0

## 💻 Cara Clone & Menjalankan

1. **Clone Repositori**
    ```bash
    git clone https://github.com/FarelNandaS/FingerTracking.git
    cd FingerTracking

2. **Buat & Aktifkan Virtual Environment**
    ```bash
    #windows (powershell)
    python -m venv env
    .\env\Scripts\activate

    #linux / macOS
    python3 -m venv env
    source env/bin/activate

3. **Install Dependensi**
    ```bash
    pip install -r requirements.txt

4. **Jalankan Program**
    ```bash
    python main.py