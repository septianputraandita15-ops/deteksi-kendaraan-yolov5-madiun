import streamlit as st
import torch
import cv2
import tempfile
import os

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Kendaraan YOLOv5",
    page_icon="🚗",
    layout="wide"
)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )
    return model

model = load_model()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.markdown("**Nama:** Septian Putra Andita")
    st.markdown("**NIM:** 143225922")
    st.caption("Deteksi Kendaraan YOLOv5 – Kota Madiun")

    conf_thres = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.5, 0.05
    )

# =========================
# JUDUL
# =========================
st.title("🚗 Deteksi Kendaraan Menggunakan YOLOv5")
st.write("Upload video lalu lintas untuk mendeteksi kendaraan.")

# =========================
# UPLOAD VIDEO
# =========================
video_file = st.file_uploader(
    "📤 Upload Video (.mp4)",
    type=["mp4"]
)

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Video Input")
        st.video(tfile.name)

    with col2:
        st.subheader("📊 Hasil Deteksi")
        st.info("Klik tombol proses untuk melihat hasil")

    if st.button("🔍 Proses Deteksi"):
        with st.spinner("⏳ Memproses video..."):
            cap = cv2.VideoCapture(tfile.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = "hasil_deteksi.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            model.conf = conf_thres

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                annotated_frame = results.render()[0]
                out.write(annotated_frame)

            cap.release()
            out.release()

        st.success("✅ Deteksi selesai")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Video Input")
            st.video(tfile.name)

        with col2:
            st.subheader("📊 Video Hasil Deteksi")
            st.video(output_path)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Aplikasi ini dikembangkan sebagai bagian dari Tugas Akhir "
    "Program Studi Teknik Informatika Universitas Maarif Hasyim Latif"
)
