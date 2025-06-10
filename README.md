## Program Pemantauan Lalu Lintas 
Program ini adalah sistem pemantauan lalu lintas yang menganalisis rekaman video (traffic_video.mp4) untuk menyediakan wawasan detail tentang pergerakan dan karakteristik kendaraan. Menggabungkan deteksi objek mutakhir dengan pelacakan yang efisien.

## Deskripsi singkat program
- Mengidentifikasi mobil, truk, dan bus menggunakan YOLOv8 dengan akurasi tinggi.
- Melacak setiap kendaraan secara konsisten di seluruh frame menggunakan DeepSORT, memberikan ID unik untuk pemantauan berkelanjutan.
- Secara otomatis menghitung kendaraan yang melintasi garis virtual, dengan mekanisme untuk menghindari penghitungan ganda.
- Menghitung dan menampilkan kecepatan kendaraan dalam km/jam.
- Menganalisis dan mengklasifikasikan warna kendaraan (misalnya, Hitam, Putih, Merah, Biru, dll.).
- Menandai dan menampilkan durasi kendaraan yang berhenti atau bergerak sangat lambat.
- Menampilkan kotak pembatas, ID pelacakan, kecepatan, warna, status idle, jejak jalur, serta ringkasan hitungan total dan distribusi warna kendaraan langsung pada video.

## Teknologi yang Digunakan
- Python 3.x
- OpenCV (cv2): Untuk pemrosesan video dan visualisasi.
- Ultralytics YOLO (ultralytics): Untuk deteksi objek.
- DeepSORT (deep_sort_realtime): Untuk pelacakan multi-objek.
- NumPy: Untuk operasi numerik.
