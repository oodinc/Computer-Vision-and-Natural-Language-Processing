# Ch. 3: Computer Vision and Natural Language Processing

## 1. Assignment Chapter 3 - Case 1

### Ringkasan Tugas

Tugas ini adalah untuk mereplikasi algoritma peningkatan gambar pada kamera smartphone, seperti teknologi *Deep Fusion* pada Apple iPhone dan *Adaptive Tetra-squared Pixel Sensor* pada Samsung Galaxy, yang bertujuan untuk menghasilkan foto yang cerah dan berkualitas tinggi meskipun dalam kondisi minim cahaya. Teknik yang diterapkan meliputi **Max Pooling** dan **Contrast Limited Adaptive Histogram Equalization (CLAHE)** untuk meningkatkan kualitas visual dari gambar gelap.

### Dataset

Tugas ini menggunakan dua gambar sebagai contoh: 
1. **photo1.jpeg**: sebagai gambar utama yang akan ditingkatkan kualitasnya.
2. **lena.png**: sebagai gambar uji untuk eksplorasi awal.

> **Catatan**: Kedua file gambar ini harus diunggah ke Google Colab sebelum menjalankan kode.

### Library yang Digunakan

- **NumPy**: Untuk operasi numerik pada matriks dan gambar.
- **OpenCV**: Untuk pengolahan gambar, seperti konversi warna dan histogram.
- **Scikit-Image**: Untuk operasi *pooling*, seperti Max Pooling, Min Pooling, dan Average Pooling.
- **Matplotlib**: Untuk visualisasi gambar dan histogram.
- **PyTorch**: Untuk Max Pooling menggunakan tensor, memungkinkan pemrosesan yang lebih efisien pada GPU.

### Persyaratan

- Semua modul yang dibutuhkan (beserta versinya) harus sudah terpasang dengan benar.
- Kode pada baris `#TODO` di dalam notebook harus dilengkapi sesuai dengan petunjuk yang diberikan.
- **DILARANG MENGUBAH** kode pada bagian *user-defined function (UDFs)* yang telah disediakan dalam tugas ini.

### Langkah Pengerjaan

1. **Eksplorasi Dasar Pengolahan Gambar Menggunakan OpenCV**:
   - Mengonversi gambar warna menjadi grayscale dan biner.
   - Memvisualisasikan histogram dari gambar warna dan grayscale.

2. **Penerapan Max Pooling**:
   - Menggunakan `block_reduce` dari Scikit-Image untuk menerapkan Max Pooling dengan blok 4x4.
   - Menerapkan Max Pooling menggunakan PyTorch untuk memanfaatkan efisiensi GPU.

3. **Eksplorasi Min Pooling dan Average Pooling**:
   - Mengimplementasikan Min Pooling dan Average Pooling pada gambar dan menganalisis perbedaan hasil antara kedua metode.

4. **Penerapan CLAHE untuk Peningkatan Gambar**:
   - Menggunakan CLAHE untuk meningkatkan kontras gambar dengan menjaga detail penting pada area gelap.

5. **Simpan Hasil Gambar Akhir**:
   - Menyimpan hasil gambar yang telah ditingkatkan menggunakan CLAHE dengan nama `Tim 3.png`.

### Hasil dan Kesimpulan

#### Hasil
- Teknik **Max Pooling** menghasilkan gambar dengan area yang lebih terang, tetapi cenderung mengorbankan beberapa detail kecil karena hanya mengambil nilai maksimum dalam area pooling.
- Teknik **Min Pooling** menghasilkan gambar yang lebih gelap, sedangkan **Average Pooling** menghasilkan gambar dengan tingkat kecerahan sedang yang mempertahankan lebih banyak detail halus.
- **CLAHE** terbukti lebih efektif dalam meningkatkan kualitas visual dari gambar gelap karena teknik ini menyesuaikan kontras dengan lebih presisi dan menjaga detail yang lebih baik dibandingkan Max Pooling.

#### Kesimpulan
Penerapan CLAHE untuk meningkatkan gambar gelap memberikan hasil yang lebih baik dibandingkan dengan Max Pooling, karena teknologi ini dirancang untuk meningkatkan kontras pada area dengan pencahayaan rendah tanpa mengorbankan detail gambar. Sementara Max Pooling lebih cocok untuk aplikasi seperti *downsampling* dalam konteks deep learning. Dalam konteks fotografi smartphone, pendekatan berbasis CLAHE lebih sesuai untuk menghasilkan gambar yang cerah dan berkualitas tinggi dalam kondisi pencahayaan yang minim.

---

## 2. Assignment Chapter 3 - Case 2

### Rangkuman Tugas
Tugas ini melibatkan pengembangan model *Computer Vision* untuk mengenali digit tulisan tangan (0-9). Karena keterbatasan waktu, pendekatan yang digunakan adalah **Transfer Learning** menggunakan model *pre-trained* yang telah tersedia di pustaka *PyTorch*, yaitu **ResNet18**, **DenseNet121**, dan **Vision Transformer (ViT)**.

Model ini dilatih untuk mengidentifikasi angka dari dataset MNIST, menggunakan *transfer learning* dengan beberapa pengaturan *layer* yang dibekukan. Hasil pelatihan dievaluasi melalui metrik akurasi dan loss pada data pelatihan dan validasi.

### Dataset
- **MNIST Handwritten Digits**: Dataset berisi gambar hitam-putih (grayscale) dengan 10 kelas (angka 0-9), yang banyak digunakan untuk tugas *image classification*.

## Library yang Digunakan
- **PyTorch**: Untuk pembuatan dan pelatihan model jaringan saraf tiruan (artificial neural network).
- **Torchvision**: Untuk mengambil dataset MNIST dan memuat beberapa model *pre-trained*.
- **Scikit-learn**: Untuk dukungan dalam visualisasi data dan evaluasi performa.
- **Matplotlib**: Untuk visualisasi hasil pelatihan dalam bentuk grafik akurasi dan loss.

### Persyaratan
- Semua modul (termasuk versi yang sesuai) harus terinstal, terutama PyTorch 2.0.1 dan Torchvision 0.15.2.
- Hardware GPU yang tersedia (direkomendasikan menggunakan Google Colab dengan GPU T4).

### Langkah Pengerjaan
1. **Modifikasi Model Pre-trained**:
   - Pilih model DenseNet sebagai eksperimen awal.
   - Sesuaikan layer pertama (input) dan layer terakhir (output) agar sesuai dengan dataset MNIST.
2. **Penentuan Hyperparameter**:
   - Tentukan nilai *batch size* dan *learning rate* yang optimal untuk pelatihan.
3. **Latihan dan Evaluasi Model**:
   - Latih model untuk beberapa epoch, dan visualisasikan akurasi serta loss pada data pelatihan dan validasi.
4. **Pembekuan (Freezing) Layer**:
   - Bekukan beberapa layer DenseNet secara bertahap: pertama pada "denseblock1" saja, lalu pada "denseblock1" dan "denseblock2".
   - Latih ulang model untuk melihat perbedaan hasil dengan pembekuan layer.
5. **BONUS**:
   - Ulangi semua langkah di atas menggunakan model lain, yaitu ResNet18 dan Vision Transformer (ViT).

### Hasil dan Kesimpulan
1. **Akurasi dan Loss**:
   - Model yang dilatih sepenuhnya (tanpa layer yang dibekukan) menunjukkan hasil akurasi terbaik.
   - Model dengan layer yang dibekukan cenderung memiliki akurasi lebih rendah karena kurang fleksibel dalam menyesuaikan representasi fitur sesuai dataset MNIST.
2. **Pengaruh Freezing Layer**:
   - Semakin banyak layer yang dibekukan, semakin rendah akurasi di awal epoch karena model mengalami kesulitan dalam beradaptasi dengan dataset baru.
   - Waktu pelatihan dan validasi lebih cepat pada model dengan layer yang dibekukan karena pengurangan perhitungan gradien dan update parameter.
3. **Kesimpulan**:
   - Pendekatan *transfer learning* memberikan efisiensi waktu pelatihan dengan akurasi yang cukup baik pada MNIST. Namun, pembekuan layer tertentu dapat mengurangi performa jika fitur yang dibekukan kurang sesuai untuk tugas klasifikasi angka pada dataset MNIST.

---

## 3. Assignment Chapter 3 - Case 3
### Ringkasan Tugas
Tugas ini bertujuan untuk mengimplementasikan deteksi objek secara *real-time* menggunakan model YOLOv5 yang telah dilatih sebelumnya, dengan memanfaatkan video dari YouTube sebagai input. Model ini akan mendeteksi objek-objek yang ada dalam video dan menggambar kotak pembatas (bounding box) serta label pada objek yang terdeteksi.

### Dataset
- **Dataset**: Video YouTube yang berisi berbagai objek yang ingin dideteksi, seperti manusia, kendaraan, dan objek lainnya.
    - Contoh URL YouTube yang digunakan:
        1. **Crowded Place**: https://www.youtube.com/watch?v=dwD1n7N7EAg
        2. **Solar System**: https://www.youtube.com/watch?v=g2KmtA97HxY
        3. **Road Traffic**: https://www.youtube.com/watch?v=wqctLW0Hb_0

### Library yang Digunakan
- **PyTorch**: Untuk membangun dan menggunakan model YOLOv5 dalam deteksi objek.
- **Numpy**: Untuk operasi numerik, terutama dalam pengolahan frame video.
- **OpenCV2**: Untuk menangani video, termasuk membaca frame dan menggambar bounding box pada gambar.
- **cap-from-youtube**: Untuk mengambil video langsung dari YouTube menggunakan URL.

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul berikut sudah terinstal:
    - PyTorch
    - OpenCV
    - cap-from-youtube
- **CUDA**: Pastikan GPU diaktifkan di Google Colab (Runtime > Change runtime type > pilih T4 GPU).
  
### Langkah Pengerjaan
1. **Instalasi dan Persiapan**:
   - Install library tambahan menggunakan pip: `!pip install cap-from-youtube`.
   - Import semua library yang diperlukan seperti PyTorch, OpenCV, dan cap-from-youtube.
   
2. **Pemanggilan Model YOLOv5**:
   - Panggil model YOLOv5 yang telah dilatih sebelumnya menggunakan PyTorch Hub dengan kode `torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)`.

3. **Pemrosesan Video**:
   - Ambil video dari YouTube menggunakan fungsi `cap_from_youtube(url)`.
   - Proses video frame-by-frame dan deteksi objek dalam setiap frame menggunakan model YOLOv5.
   - Gambar bounding box dan label untuk setiap objek yang terdeteksi.

4. **Evaluasi Deteksi Objek**:
   - Menilai akurasi deteksi berdasarkan video yang diuji. Menghitung Frames Per Second (FPS) untuk evaluasi kinerja model dalam deteksi objek secara real-time.

5. **Eksperimen dengan Video yang Berbeda**:
   - Uji deteksi objek dengan berbagai jenis video dari YouTube (misalnya video tentang keramaian, sistem tata surya, dan lalu lintas jalanan).

### Hasil dan Kesimpulan
- **Hasil Deteksi**:
    - Model YOLOv5 dapat mendeteksi objek dengan baik pada video berisi keramaian dan lalu lintas jalanan. Namun, pada video dengan objek animasi seperti tata surya, akurasi deteksi menjadi buruk.
    - Pada video animasi (seperti solar system), model kesulitan mendeteksi objek dengan benar karena latar belakang dan pewarnaan objek yang berbeda dengan data pelatihan.

- **FPS (Frames Per Second)**:
    - FPS dihitung selama pemrosesan video dan menunjukkan kinerja deteksi objek dalam waktu nyata. FPS yang lebih tinggi menandakan model lebih efisien dalam mendeteksi objek secara real-time.

- **Kesimpulan**:
    - Meskipun YOLOv5 cukup akurat untuk deteksi objek pada video yang lebih realistis, model ini memiliki keterbatasan dalam mendeteksi objek di video dengan latar belakang atau warna yang sangat berbeda dari data pelatihan (seperti video animasi).

---

## 4. Assignment Chapter 3 - Case 4
### Ringkasan Tugas
Tugas ini bertujuan untuk mengklasifikasikan tweet terkait bencana menggunakan model pre-trained BERT. Model BERT diterapkan untuk menganalisis apakah sebuah tweet berisi informasi terkait bencana atau bukan. Melalui teknik fine-tuning pada model BERT, tugas ini diharapkan dapat digunakan untuk:
- Menganalisis pola komunikasi masyarakat dalam situasi darurat.
- Mengembangkan sistem peringatan dini berbasis Twitter.
- Membantu organisasi bantuan bencana dalam respons terhadap kejadian darurat.

### Dataset
Dataset yang digunakan dalam tugas ini adalah *Disaster Tweets* yang berisi tweet yang dapat berkaitan atau tidak berkaitan dengan bencana. Dataset ini berformat CSV dengan dua kolom utama:
- **text**: Tweet yang diposting di Twitter.
- **target**: Label klasifikasi, dengan 1 untuk tweet terkait bencana dan 0 untuk tweet non-bencana.

### Library yang Digunakan
- **Pandas**: Untuk manipulasi data.
- **Numpy**: Untuk operasi numerik.
- **NLTK**: Untuk proses pembersihan teks seperti penghapusan stopwords dan emoji.
- **Scikit-learn**: Untuk pembagian data menjadi set pelatihan dan pengujian.
- **PyTorch**: Untuk membangun model dan pelatihan menggunakan framework deep learning.
- **Transformers (Hugging Face)**: Untuk mengimpor dan mengoptimalkan model BERT dalam tugas klasifikasi teks.

### Persyaratan
- **Modul dan Versi**: Pastikan semua modul yang diperlukan sudah terinstall, termasuk `transformers`, `torch`, `pandas`, `numpy`, dan `nltk`.
- **GPU**: Gunakan GPU (CUDA) untuk akselerasi pelatihan model.
- **Dataset**: Dataset harus tersedia dalam format CSV yang sesuai dengan spesifikasi kolom `text` dan `target`.

### Langkah Pengerjaan
1. **Pembersihan Data (Preprocessing)**: 
   - Menghilangkan URL, HTML tags, dan emoji dari teks.
   - Menghapus tanda baca dan stopwords menggunakan NLTK.
   
2. **Tokenisasi dengan BERT Tokenizer**: 
   - Menggunakan tokenizer BERT untuk mengonversi tweet menjadi token yang dapat diproses oleh model.

3. **Pembagian Dataset**: 
   - Membagi dataset menjadi 80% untuk pelatihan dan 20% untuk validasi.

4. **Model BERT**: 
   - Memuat model BERT pre-trained dan menyesuaikannya dengan data tugas ini menggunakan teknik fine-tuning.

5. **Pelatihan dan Validasi**: 
   - Melatih model selama beberapa epoch, mengoptimalkan hyperparameters seperti learning rate dan batch size.
   - Mengukur akurasi model pada dataset validasi.

6. **Evaluasi Akhir**: 
   - Menguji model pada data pengujian (test) dan menghasilkan prediksi apakah tweet berisi informasi bencana.

### Hasil dan Kesimpulan
Setelah melatih model BERT pada data pelatihan dan melakukan validasi, model yang terlatih berhasil mencapai akurasi di atas 80%. Model BERT yang telah dilatih disimpan dan digunakan untuk melakukan prediksi pada data uji. Berdasarkan hasil evaluasi, model berhasil memprediksi tweet dengan baik dan menunjukkan kemampuan untuk mengklasifikasikan tweet terkait bencana dengan akurasi tinggi.

Kesimpulannya, fine-tuning BERT untuk klasifikasi teks pada dataset bencana memberikan hasil yang baik, dengan potensi untuk digunakan dalam sistem peringatan dini berbasis media sosial.

