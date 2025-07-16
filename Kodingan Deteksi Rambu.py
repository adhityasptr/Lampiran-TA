#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
import threading

class SistemLaluLintasLengkapOptimal:
    def _init_(self):
        rospy.init_node('sistem_lalu_lintas_lengkap_optimal', anonymous=True)
        
        model_path = '/data/models/best.pt'
        rospy.loginfo("🔄 Memuat model YOLOv8...")
        self.model = YOLO(model_path)
        
        self.model.overrides['verbose'] = False
        self.model.overrides['save'] = False
        self.model.overrides['show'] = False
        
        rospy.loginfo("✅ Model YOLOv8 berhasil dimuat dengan optimasi!")
        rospy.loginfo(f"📊 Kelas model: {self.model.names}")
        
        self.aturan_lalu_lintas = {
            0: {
                'nama': 'Dilarang Belok Kanan',
                'pengenalan': 'Tidak boleh belok kanan - BELOK KIRI sekarang',
                'aksi': 'PAKSA_BELOK_KIRI',
                'prioritas': 'tinggi',
                'durasi': 3.0,
                'gerakan': {'v': 0.1, 'omega': 1.0}
            },
            1: {
                'nama': 'Dilarang Belok Kiri',
                'pengenalan': 'Tidak boleh belok kiri - BELOK KANAN sekarang', 
                'aksi': 'PAKSA_BELOK_KANAN',
                'prioritas': 'tinggi',
                'durasi': 3.0,
                'gerakan': {'v': 0.1, 'omega': -1.0}
            },
            2: {
                'nama': 'Dilarang Masuk',
                'pengenalan': 'Dilarang masuk - berhenti 5 detik lalu lanjut',
                'aksi': 'BERHENTI_SINGKAT',
                'prioritas': 'kritis',
                'durasi': 5.0,
                'gerakan': {'v': 0.0, 'omega': 0.0}
            },
            3: {
                'nama': 'Lampu Hijau',
                'pengenalan': 'Lampu hijau - boleh jalan dan keluar dari kondisi merah',
                'aksi': 'LAMPU_HIJAU_JALAN',
                'prioritas': 'lampu_lalu_lintas',
                'durasi': 0.5,
                'gerakan': {'v': 0.1, 'omega': 0.0}
            },
            4: {
                'nama': 'Lampu Kuning',
                'pengenalan': 'Lampu kuning - kurangi kecepatan 5 detik atau keluar dari kondisi merah',
                'aksi': 'LAMPU_KUNING_HATI_HATI',
                'prioritas': 'lampu_lalu_lintas',
                'durasi': 5.0,
                'gerakan': {'v': 0.5, 'omega': 0.0}
            },
            5: {
                'nama': 'Lampu Merah',
                'pengenalan': 'Lampu merah - BERHENTI sampai hijau/kuning terdeteksi',
                'aksi': 'LAMPU_MERAH_BERHENTI',
                'prioritas': 'lampu_lalu_lintas',
                'durasi': 999.0,
                'gerakan': {'v': 0.0, 'omega': 0.0}
            },
            6: {
                'nama': 'Rambu Berhenti',
                'pengenalan': 'Rambu stop - berhenti 20 detik lalu lanjut',
                'aksi': 'BERHENTI_LALU_JALAN',
                'prioritas': 'kritis',
                'durasi': 20.0,
                'gerakan': {'v': 0.0, 'omega': 0.0}
            },
            7: {
                'nama': 'Rambu Penyebrangan',
                'pengenalan': 'Area penyebrangan - jalan pelan 5 detik',
                'aksi': 'MAJU_PELAN',
                'prioritas': 'tinggi',
                'durasi': 5.0,
                'gerakan': {'v': 0.5, 'omega': 0.0}
            }
        }
        
        self.mengikuti_jalur_aktif = True
        self.override_lalu_lintas = False
        self.aksi_sekarang = None
        self.waktu_mulai_override = 0
        self.durasi_override = 0
        self.pengenalan_terakhir = {}
        self.cooldown_pengenalan = 2.0
        self.kecepatan_maju_default = 0.1
        
        self.status_lampu_lalu_lintas = 'TIDAK_DIKETAHUI'
        self.waktu_deteksi_lampu_merah = 0
        self.menunggu_di_lampu_merah = False
        
        self.ambang_kepercayaan = 0.60
        self.ukuran_input_deteksi = 160
        self.ukuran_input_tampilan = 480
        self.fps_maksimal = 0.8
        self.waktu_proses_terakhir = 0
        self.kunci_pemrosesan = threading.Lock()
        
        self.jumlah_frame = 0
        self.total_waktu_proses = 0
        self.jumlah_deteksi = 0
        
        self.detection_pub = rospy.Publisher('/duckduck/traffic_detections', String, queue_size=1)
        self.recognition_pub = rospy.Publisher('/duckduck/traffic_recognition', String, queue_size=1)
        self.status_pub = rospy.Publisher('/duckduck/system_status', String, queue_size=1)
        self.image_pub = rospy.Publisher('/duckduck/annotated_image/compressed', 
                                       CompressedImage, queue_size=1)
        
        self.penerbit_mobil = []
        
        try:
            pub1 = rospy.Publisher('/duckduck/joy_mapper_node/car_cmd', Twist, queue_size=1)
            self.penerbit_mobil.append(('joy_mapper', pub1, 'Twist'))
            rospy.loginfo("✅ Menambahkan penerbit joy_mapper")
        except:
            pass
            
        try:
            pub2 = rospy.Publisher('/duckduck/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
            self.penerbit_mobil.append(('car_cmd_switch', pub2, 'Twist2DStamped'))
            rospy.loginfo("✅ Menambahkan penerbit car_cmd_switch")
        except:
            pass
            
        try:
            pub3 = rospy.Publisher('/duckduck/lane_controller_node/switch', BoolStamped, queue_size=1)
            self.lane_switch_pub = pub3
            rospy.loginfo("✅ Menambahkan pengontrol jalur")
        except:
            self.lane_switch_pub = None
        
        self.image_sub = rospy.Subscriber('/duckduck/camera_node/image/compressed', 
                                        CompressedImage, self.callback_gambar, queue_size=1)
        
        self.timer_kontrol = rospy.Timer(rospy.Duration(0.2), self.loop_kontrol)
        
        self.timer_performa = rospy.Timer(rospy.Duration(10.0), self.log_performa)
        
        rospy.Timer(rospy.Duration(2.0), self.mulai_maju_default, oneshot=True)
        
        rospy.loginfo("🚀 Sistem Lalu Lintas Pengenalan Langsung telah diinisialisasi!")
        rospy.loginfo(f"🎯 Pengenalan Langsung: Deteksi {self.ukuran_input_deteksi}px, Tampilan {self.ukuran_input_tampilan}px, Ambang {self.ambang_kepercayaan}, {len(self.penerbit_mobil)} penerbit")
        self.log_aturan_lalu_lintas()
        
    def mulai_maju_default(self, event=None):
        rospy.loginfo("🚗 Memulai gerakan maju yang dioptimalkan")
        self.kirim_perintah_mobil(self.kecepatan_maju_default, 0.0)
        
    def log_aturan_lalu_lintas(self):
        rospy.loginfo("📋 Aturan Lalu Lintas PENGENALAN LANGSUNG:")
        rospy.loginfo("   🔴 Lampu merah → BERHENTI sampai hijau/kuning terdeteksi")
        rospy.loginfo("   🟢 Lampu hijau → 0.25 m/s (0.90 km/h)")
        rospy.loginfo("   🟡 Lampu kuning → 0.10 m/s (0.36 km/h) selama 5 detik")
        rospy.loginfo("   ↰ Dilarang kanan → BELOK KIRI (ω=0.8) selama 4 detik")
        rospy.loginfo("   ↱ Dilarang kiri → BELOK KANAN (ω=-0.8) selama 4 detik")
        rospy.loginfo("   🚫 Dilarang masuk → 5.0 detik")
        rospy.loginfo("   🛑 Rambu stop → 20.0 detik")
        rospy.loginfo("   🚶 Penyebrangan → 0.15 m/s (0.54 km/h)")
        rospy.loginfo("   ⚡ PEMICU: kepercayaan ≥ 0.60 (sebelumnya 0.80)")
    
    def callback_gambar(self, msg):
        if not self.kunci_pemrosesan.acquire(blocking=False):
            return
        
        try:
            waktu_sekarang = time.time()
            if waktu_sekarang - self.waktu_proses_terakhir < (1.0 / self.fps_maksimal):
                return
            
            waktu_mulai = time.time()
            
            np_arr = np.frombuffer(msg.data, np.uint8)
            gambar_asli = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if gambar_asli is None:
                return
            
            tinggi, lebar = gambar_asli.shape[:2]
            if lebar > self.ukuran_input_deteksi:
                skala = self.ukuran_input_deteksi / lebar
                lebar_baru = self.ukuran_input_deteksi
                tinggi_baru = int(tinggi * skala)
                gambar_deteksi = cv2.resize(gambar_asli, (lebar_baru, tinggi_baru), interpolation=cv2.INTER_LINEAR)
            else:
                gambar_deteksi = gambar_asli.copy()
            
            gambar_tampilan = gambar_asli.copy()
            if self.jumlah_frame % 2 == 0:
                if lebar > self.ukuran_input_tampilan:
                    skala = self.ukuran_input_tampilan / lebar
                    lebar_baru = self.ukuran_input_tampilan
                    tinggi_baru = int(tinggi * skala)
                    gambar_tampilan = cv2.resize(gambar_asli, (lebar_baru, tinggi_baru), interpolation=cv2.INTER_CUBIC)
            
            hasil = self.model(
                gambar_deteksi,
                conf=self.ambang_kepercayaan,
                verbose=False,
                device='cpu',
                half=False,
                augment=False,
                agnostic_nms=False,
                max_det=10
            )
            
            deteksi = self.proses_deteksi_dan_aksi(hasil[0], waktu_sekarang, gambar_deteksi.shape, gambar_asli.shape)
            
            if deteksi:
                self.terbitkan_deteksi_async(deteksi)
            
            self.jumlah_frame += 1
            if self.jumlah_frame % 2 == 0:
                self.terbitkan_gambar_teranotasi_berkualitas_tinggi(hasil[0], gambar_tampilan, msg.header, gambar_deteksi.shape)
            
            waktu_proses = time.time() - waktu_mulai
            self.total_waktu_proses += waktu_proses
            
            if waktu_proses > 2.0:
                rospy.logwarn(f"⚠ Pemrosesan lambat: {waktu_proses:.2f} detik")
            
            self.waktu_proses_terakhir = waktu_sekarang
            
        except Exception as e:
            rospy.logerr(f"❌ Kesalahan dalam pemrosesan gambar: {e}")
        finally:
            self.kunci_pemrosesan.release()
    
    def proses_deteksi_dan_aksi(self, hasil, waktu_sekarang, bentuk_deteksi, bentuk_asli):
        deteksi = []
        
        if hasil.boxes is not None and len(hasil.boxes) > 0:
            kotak = hasil.boxes.xyxy.cpu().numpy()
            kepercayaan = hasil.boxes.conf.cpu().numpy()
            kelas = hasil.boxes.cls.cpu().numpy().astype(int)
            
            skala_x = bentuk_asli[1] / bentuk_deteksi[1]
            skala_y = bentuk_asli[0] / bentuk_deteksi[0]
            
            for i in range(len(kotak)):
                id_kelas = kelas[i]
                nilai_kepercayaan = kepercayaan[i]
                
                if nilai_kepercayaan >= self.ambang_kepercayaan:
                    aturan_lalu_lintas = self.aturan_lalu_lintas.get(id_kelas, {})
                    nama_kelas = aturan_lalu_lintas.get('nama', f'kelas_{id_kelas}')
                    
                    kotak_terskal = [
                        float(kotak[i][0] * skala_x),
                        float(kotak[i][1] * skala_y),
                        float(kotak[i][2] * skala_x),
                        float(kotak[i][3] * skala_y)
                    ]
                    
                    deteksi_item = {
                        'id_kelas': int(id_kelas),
                        'nama_kelas': nama_kelas,
                        'kepercayaan': float(nilai_kepercayaan),
                        'kotak': kotak_terskal,
                        'waktu': waktu_sekarang,
                        'aturan_lalu_lintas': aturan_lalu_lintas
                    }
                    
                    deteksi.append(deteksi_item)
                    self.jumlah_deteksi += 1
                    
                    if nilai_kepercayaan >= 0.60 and aturan_lalu_lintas:
                        self.eksekusi_aksi_lalu_lintas_optimal(id_kelas, aturan_lalu_lintas, nilai_kepercayaan, waktu_sekarang)
        
        return deteksi
    
    def eksekusi_aksi_lalu_lintas_optimal(self, id_kelas, aturan_lalu_lintas, kepercayaan, waktu_sekarang):
        waktu_terakhir = self.pengenalan_terakhir.get(id_kelas, 0)
        if waktu_sekarang - waktu_terakhir < self.cooldown_pengenalan:
            return
        
        aksi = aturan_lalu_lintas['aksi']
        nama_kelas = aturan_lalu_lintas['nama']
        pengenalan = aturan_lalu_lintas['pengenalan']
        prioritas = aturan_lalu_lintas['prioritas']
        durasi = aturan_lalu_lintas['durasi']
        gerakan = aturan_lalu_lintas['gerakan']
        
        if prioritas == 'lampu_lalu_lintas':
            self.tangani_deteksi_lampu_lalu_lintas(id_kelas, aksi, nama_kelas, pengenalan, waktu_sekarang)
            self.pengenalan_terakhir[id_kelas] = waktu_sekarang
            return
        
        if self.override_lalu_lintas and (waktu_sekarang - self.waktu_mulai_override) < self.durasi_override:
            return
        
        if self.menunggu_di_lampu_merah:
            rospy.logwarn(f"⏸ Menunggu di lampu merah - mengabaikan {nama_kelas}")
            return
        
        rospy.logwarn(f"🎯 PENGENALAN: {nama_kelas}")
        rospy.logwarn(f"💭 BERPIKIR: {pengenalan}")
        rospy.logwarn(f"⚡ AKSI: {aksi} (prioritas: {prioritas}, durasi: {durasi} detik)")
        
        self.override_lalu_lintas = True
        self.aksi_sekarang = aksi
        self.waktu_mulai_override = waktu_sekarang
        self.durasi_override = durasi
        
        if aksi == 'PAKSA_BELOK_KIRI':
            self.eksekusi_belok(gerakan, 'KIRI', durasi, nama_kelas)
            
        elif aksi == 'PAKSA_BELOK_KANAN':
            self.eksekusi_belok(gerakan, 'KANAN', durasi, nama_kelas)
            
        elif aksi == 'BERHENTI_SINGKAT':
            self.eksekusi_berhenti_singkat(durasi, nama_kelas)
            
        elif aksi == 'BERHENTI_LALU_JALAN':
            self.eksekusi_berhenti_lalu_jalan(durasi, nama_kelas)
            
        elif aksi == 'MAJU_PELAN':
            self.eksekusi_maju_pelan(gerakan, durasi, nama_kelas)
        
        self.pengenalan_terakhir[id_kelas] = waktu_sekarang
        
        self.terbitkan_pengenalan_async(nama_kelas, aksi, pengenalan, kepercayaan)
    
    def tangani_deteksi_lampu_lalu_lintas(self, id_kelas, aksi, nama_kelas, pengenalan, waktu_sekarang):
        aturan_lalu_lintas = self.aturan_lalu_lintas.get(id_kelas, {})
        gerakan = aturan_lalu_lintas.get('gerakan', {'v': 0.3, 'omega': 0.0})
        
        if aksi == 'LAMPU_MERAH_BERHENTI':
            if not self.menunggu_di_lampu_merah:
                rospy.logwarn(f"🔴 LAMPU MERAH TERDETEKSI: {nama_kelas}")
                rospy.logwarn(f"💭 BERPIKIR: {pengenalan}")
                rospy.logwarn(f"🛑 BERHENTI sampai HIJAU/KUNING terdeteksi!")
                
                self.status_lampu_lalu_lintas = 'MERAH_BERHENTI'
                self.menunggu_di_lampu_merah = True
                self.waktu_deteksi_lampu_merah = waktu_sekarang
                
                self.override_lalu_lintas = True
                self.aksi_sekarang = aksi
                self.waktu_mulai_override = waktu_sekarang
                self.durasi_override = 999.0
                
                self.atur_mengikuti_jalur(False)
                self.kirim_perintah_mobil(0.0, 0.0)
                
                self.terbitkan_pengenalan_async(nama_kelas, aksi, pengenalan, 1.0)
                self.terbitkan_status_async('LAMPU_MERAH_BERHENTI', 'Menunggu hijau/kuning')
            
            else:
                berlalu = waktu_sekarang - self.waktu_deteksi_lampu_merah
                rospy.loginfo(f"🔴 Masih menunggu di lampu merah ({berlalu:.1f} detik)")
        
        elif aksi in ['LAMPU_HIJAU_JALAN', 'LAMPU_KUNING_HATI_HATI']:
            if self.menunggu_di_lampu_merah:
                berlalu = waktu_sekarang - self.waktu_deteksi_lampu_merah
                rospy.logwarn(f"🟢 {nama_kelas.upper()} TERDETEKSI!")
                rospy.logwarn(f"💭 BERPIKIR: {pengenalan}")
                rospy.logwarn(f"✅ MELANJUTKAN setelah {berlalu:.1f} detik di lampu merah")
                
                self.status_lampu_lalu_lintas = 'HIJAU_JALAN' if aksi == 'LAMPU_HIJAU_JALAN' else 'KUNING_HATI_HATI'
                self.menunggu_di_lampu_merah = False
                self.waktu_deteksi_lampu_merah = 0
                
                if aksi == 'LAMPU_KUNING_HATI_HATI':
                    rospy.logwarn(f"🟡 LAMPU KUNING: Mengurangi kecepatan selama 5 detik")
                    self.override_lalu_lintas = True
                    self.aksi_sekarang = aksi
                    self.waktu_mulai_override = waktu_sekarang
                    self.durasi_override = 5.0
                    self.atur_mengikuti_jalur(False)
                    self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
                else:
                    self.override_lalu_lintas = False
                    self.aksi_sekarang = None
                    self.atur_mengikuti_jalur(True)
                    self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
                
                self.terbitkan_pengenalan_async(nama_kelas, aksi, pengenalan, 1.0)
                self.terbitkan_status_async('DILANJUTKAN_DARI_MERAH', f'{nama_kelas} terdeteksi setelah {berlalu:.1f} detik')
            
            else:
                if aksi == 'LAMPU_KUNING_HATI_HATI':
                    rospy.logwarn(f"🟡 LAMPU KUNING: Mengurangi kecepatan selama 5 detik")
                    self.status_lampu_lalu_lintas = 'KUNING_HATI_HATI'
                    
                    self.override_lalu_lintas = True
                    self.aksi_sekarang = aksi
                    self.waktu_mulai_override = waktu_sekarang
                    self.durasi_override = 5.0
                    
                    self.atur_mengikuti_jalur(False)
                    self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
                    
                    self.terbitkan_pengenalan_async(nama_kelas, aksi, pengenalan, 1.0)
                    self.terbitkan_status_async('KUNING_KECEPATAN_PELAN', 'Mengurangi kecepatan selama 5 detik')
                    
                else:
                    rospy.loginfo(f"🟢 {nama_kelas} terdeteksi - melanjutkan gerakan normal")
                    self.status_lampu_lalu_lintas = 'HIJAU_JALAN'
                    self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
                    self.terbitkan_pengenalan_async(nama_kelas, aksi, pengenalan, 1.0)
    
    def eksekusi_belok(self, gerakan, arah, durasi, nama_kelas):
        rospy.logwarn(f"🔄 MELAKUKAN BELOKAN {arah} selama {durasi} detik karena {nama_kelas}")
        rospy.logwarn(f"🎯 BERBELOK: v={gerakan['v']}, ω={gerakan['omega']}")
        
        self.atur_mengikuti_jalur(False)
        
        self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
        
        self.terbitkan_status_async(f'SEDANG_BERBELOK_{arah}', f'Belokan {arah.lower()} karena {nama_kelas}')
    
    def kirim_perintah_mobil_smooth(self, v_target, omega_target, durasi, arah):
        """Fungsi ini tidak digunakan lagi - menggunakan kirim_perintah_mobil biasa"""
        pass
    
    def eksekusi_berhenti_singkat(self, durasi, nama_kelas):
        rospy.logwarn(f"🛑 BERHENTI SINGKAT selama {durasi} detik karena {nama_kelas}")
        
        self.atur_mengikuti_jalur(False)
        self.kirim_perintah_mobil(0.0, 0.0)
        
        self.terbitkan_status_async('BERHENTI_SINGKAT', nama_kelas)
    
    def eksekusi_lanjut_normal(self, gerakan, nama_kelas):
        rospy.loginfo(f"✅ LANJUT NORMAL karena {nama_kelas}")
        
        self.override_lalu_lintas = False
        self.aksi_sekarang = None
        self.atur_mengikuti_jalur(True)
        
        self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
        self.terbitkan_status_async('MELANJUTKAN', nama_kelas)
    
    def eksekusi_kurangi_kecepatan(self, gerakan, durasi, nama_kelas):
        rospy.logwarn(f"🐌 KURANGI KECEPATAN selama {durasi} detik karena {nama_kelas}")
        
        self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
        self.terbitkan_status_async('PELAN', nama_kelas)
    
    def eksekusi_berhenti_sementara(self, gerakan, durasi, nama_kelas):
        rospy.logwarn(f"🛑 BERHENTI SEMENTARA selama {durasi} detik karena {nama_kelas}")
        
        self.atur_mengikuti_jalur(False)
        self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
        
        self.terbitkan_status_async('BERHENTI_SEMENTARA', nama_kelas)
    
    def eksekusi_berhenti_lalu_jalan(self, durasi, nama_kelas):
        rospy.logwarn(f"⏸ BERHENTI LALU JALAN selama {durasi} detik karena {nama_kelas}")
        
        self.atur_mengikuti_jalur(False)
        self.kirim_perintah_mobil(0.0, 0.0)
        
        self.terbitkan_status_async('BERHENTI_LALU_JALAN', nama_kelas)
    
    def eksekusi_maju_pelan(self, gerakan, durasi, nama_kelas):
        rospy.loginfo(f"🚶 MAJU PELAN selama {durasi} detik karena {nama_kelas}")
        
        self.kirim_perintah_mobil(gerakan['v'], gerakan['omega'])
        self.terbitkan_status_async('MAJU_PELAN', nama_kelas)
    
    def atur_mengikuti_jalur(self, aktifkan):
        if self.lane_switch_pub:
            try:
                msg = BoolStamped()
                msg.header.stamp = rospy.Time.now()
                msg.data = aktifkan
                self.lane_switch_pub.publish(msg)
                
                self.mengikuti_jalur_aktif = aktifkan
                status = "DIAKTIFKAN" if aktifkan else "DINONAKTIFKAN"
                rospy.loginfo(f"🛣 Mengikuti jalur {status}")
            except:
                pass
    
    def kirim_perintah_mobil(self, kecepatan_linear, kecepatan_angular):
        for nama_pub, penerbit, tipe_msg in self.penerbit_mobil:
            try:
                if tipe_msg == 'Twist2DStamped':
                    msg = Twist2DStamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.v = kecepatan_linear
                    msg.omega = kecepatan_angular
                else:
                    msg = Twist()
                    msg.linear.x = kecepatan_linear
                    msg.angular.z = kecepatan_angular
                
                penerbit.publish(msg)
                
            except Exception as e:
                rospy.logwarn(f"Gagal menerbitkan ke {nama_pub}: {e}")
    
    def loop_kontrol(self, event):
        if self.menunggu_di_lampu_merah:
            self.kirim_perintah_mobil(0.0, 0.0)
            return
        
        if not self.override_lalu_lintas:
            self.kirim_perintah_mobil(self.kecepatan_maju_default, 0.0)
            return
        
        waktu_sekarang = time.time()
        berlalu = waktu_sekarang - self.waktu_mulai_override
        
        if berlalu >= self.durasi_override:
            rospy.loginfo(f"⏰ Aksi lalu lintas selesai setelah {berlalu:.1f} detik")
            
            self.override_lalu_lintas = False
            self.aksi_sekarang = None
            self.atur_mengikuti_jalur(True)
            self.kirim_perintah_mobil(self.kecepatan_maju_default, 0.0)
            
            self.terbitkan_status_async('DILANJUTKAN', 'Waktu aksi habis')
            rospy.loginfo("✅ Melanjutkan gerakan maju normal")
    
    def log_performa(self, event):
        if self.jumlah_frame > 0:
            rata_rata_waktu_proses = self.total_waktu_proses / self.jumlah_frame
            fps = self.jumlah_frame / 10.0
            
            rospy.loginfo(f"📊 Performa: {fps:.1f} FPS, {rata_rata_waktu_proses:.2f} detik/frame rata-rata, {self.jumlah_deteksi} deteksi")
            
            self.jumlah_frame = 0
            self.total_waktu_proses = 0
            self.jumlah_deteksi = 0
    
    def terbitkan_deteksi_async(self, deteksi):
        try:
            pesan_deteksi = String()
            pesan_deteksi.data = json.dumps(deteksi, indent=2)
            self.detection_pub.publish(pesan_deteksi)
        except:
            pass
    
    def terbitkan_pengenalan_async(self, nama_kelas, aksi, pengenalan, kepercayaan):
        try:
            data_pengenalan = {
                'waktu': time.time(),
                'nama_kelas': nama_kelas,
                'aksi': aksi,
                'pengenalan': pengenalan,
                'kepercayaan': kepercayaan
            }
            
            pesan_pengenalan = String()
            pesan_pengenalan.data = json.dumps(data_pengenalan, indent=2)
            self.recognition_pub.publish(pesan_pengenalan)
        except:
            pass
    
    def terbitkan_status_async(self, status, alasan):
        try:
            data_status = {
                'waktu': time.time(),
                'status': status,
                'alasan': alasan,
                'mengikuti_jalur': self.mengikuti_jalur_aktif,
                'override_lalu_lintas': self.override_lalu_lintas,
                'aksi_sekarang': self.aksi_sekarang
            }
            
            pesan_status = String()
            pesan_status.data = json.dumps(data_status, indent=2)
            self.status_pub.publish(pesan_status)
        except:
            pass
    
    def terbitkan_gambar_teranotasi_berkualitas_tinggi(self, hasil, gambar_tampilan, header, bentuk_deteksi):
        try:
            gambar_teranotasi = gambar_tampilan.copy()
            
            skala_x = gambar_tampilan.shape[1] / bentuk_deteksi[1]
            skala_y = gambar_tampilan.shape[0] / bentuk_deteksi[0]
            
            if hasil.boxes is not None and len(hasil.boxes) > 0:
                kotak = hasil.boxes.xyxy.cpu().numpy()
                kepercayaan = hasil.boxes.conf.cpu().numpy()
                kelas = hasil.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(kotak)):
                    if kepercayaan[i] >= self.ambang_kepercayaan:
                        x1 = int(kotak[i][0] * skala_x)
                        y1 = int(kotak[i][1] * skala_y)
                        x2 = int(kotak[i][2] * skala_x)
                        y2 = int(kotak[i][3] * skala_y)
                        
                        id_kelas = kelas[i]
                        nilai_kepercayaan = kepercayaan[i]
                        
                        aturan_lalu_lintas = self.aturan_lalu_lintas.get(id_kelas, {})
                        nama_kelas = aturan_lalu_lintas.get('nama', f'kelas_{id_kelas}')
                        aksi = aturan_lalu_lintas.get('aksi', 'TIDAK_ADA_AKSI')
                        durasi = aturan_lalu_lintas.get('durasi', 0)
                        prioritas = aturan_lalu_lintas.get('prioritas', 'rendah')
                        
                        if prioritas == 'kritis':
                            warna = (0, 0, 255)
                        elif prioritas == 'tinggi':
                            warna = (0, 140, 255)
                        elif prioritas == 'lampu_lalu_lintas':
                            warna = (255, 0, 255)
                        else:
                            warna = (0, 255, 0)
                        
                        cv2.rectangle(gambar_teranotasi, (x1, y1), (x2, y2), warna, 3)
                        
                        skala_font = 0.6
                        ketebalan = 2
                        
                        label1 = f'{nama_kelas}: {nilai_kepercayaan:.2f}'
                        
                        (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, skala_font, ketebalan)
                        
                        cv2.rectangle(gambar_teranotasi, (x1, y1-h1-10), (x1+w1+10, y1), (0, 0, 0), -1)
                        
                        cv2.putText(gambar_teranotasi, label1, (x1+5, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, skala_font, (255, 255, 255), ketebalan)
            
            if self.menunggu_di_lampu_merah:
                berlalu = time.time() - self.waktu_deteksi_lampu_merah
                teks_status = f"LAMPU MERAH - MENUNGGU ({berlalu:.1f} detik)"
                warna_status = (0, 0, 255)
            else:
                teks_status = f"Status: {self.status_lampu_lalu_lintas} | Aksi: {self.aksi_sekarang or 'MAJU'}"
                warna_status = (255, 255, 255)
                if self.override_lalu_lintas:
                    berlalu = time.time() - self.waktu_mulai_override
                    sisa = max(0, self.durasi_override - berlalu)
                    teks_status += f" ({sisa:.1f} detik tersisa)"
            
            (w, h), _ = cv2.getTextSize(teks_status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(gambar_teranotasi, (5, 5), (w+15, h+15), (0, 0, 0), -1)
            
            cv2.putText(gambar_teranotasi, teks_status, (10, h+10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_status, 2)
            
            _, buffer = cv2.imencode('.jpg', gambar_teranotasi, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            pesan_terkompresi = CompressedImage()
            pesan_terkompresi.header = header
            pesan_terkompresi.format = "jpeg"
            pesan_terkompresi.data = buffer.tobytes()
            
            self.image_pub.publish(pesan_terkompresi)
            
        except Exception as e:
            rospy.logerr(f"❌ Kesalahan dalam menerbitkan gambar berkualitas tinggi: {e}")
    
    def jalankan(self):
        rospy.loginfo("🚀 Sistem Lalu Lintas Pengenalan Langsung sedang berjalan!")
        rospy.loginfo("🎯 Fitur PENGENALAN LANGSUNG:")
        rospy.loginfo(f"   🔍 Deteksi: {self.ukuran_input_deteksi}px (pemrosesan ultra cepat)")
        rospy.loginfo(f"   📺 Tampilan: {self.ukuran_input_tampilan}px (tampilan rqt berkualitas tinggi)")
        rospy.loginfo("   📸 Kualitas gambar: 85% (jernih & tajam)")
        rospy.loginfo("   🎨 Ditingkatkan: kotak tebal, teks besar, latar belakang")
        rospy.loginfo("   ⚡ Pengenalan langsung: kepercayaan ≥ 0.60 (sebelumnya 0.80)")
        rospy.loginfo("   🔴 Lampu merah: BERHENTI sampai hijau/kuning terdeteksi")
        rospy.loginfo("   ↰ Dilarang kanan → BELOK KIRI (ω=0.8, 4 detik)")
        rospy.loginfo("   ↱ Dilarang kiri → BELOK KANAN (ω=-0.8, 4 detik)")
        rospy.spin()

if _name_ == '_main_':
    try:
        sistem = SistemLaluLintasLengkapOptimal()
        sistem.jalankan()
    except rospy.ROSInterruptException:
        rospy.loginfo("Sistem Lalu Lintas Lengkap Optimal sedang dimatikan...")
    except Exception as e:
        rospy.logerr(f"❌ Kesalahan: {e}")