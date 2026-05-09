import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.clip_model import CLIPFeatureExtractor
from src.utils.vector_db import VectorDB

st.set_page_config(page_title="Text & Image Retrieval", layout="wide")

@st.cache_resource
def load_models():
    extractor = CLIPFeatureExtractor()
    db = VectorDB()
    return extractor, db

st.title("🔍 Semantic Image Retrieval")
st.markdown("Hệ thống truy xuất hình ảnh sử dụng **CLIP** (OpenAI) và **ChromaDB**.")

extractor, db = load_models()

tab1, tab2 = st.tabs(["💬 Text-to-Image", "🖼️ Image-to-Image"])

with tab1:
    st.header("Tìm ảnh bằng Văn bản")
    text_query = st.text_input("Nhập mô tả hình ảnh (Tiếng Anh, vd: 'a lion', 'a red apple'):")
    if st.button("Tìm kiếm Text"):
        if text_query:
            with st.spinner("Đang tìm kiếm..."):
                # Áp dụng Prompt Engineering để CLIP hiểu ngữ cảnh tốt hơn
                prompt = f"a photo of a {text_query}"
                emb = extractor.get_text_embedding(prompt)
                results = db.search(emb, n_results=5)
                
                if results and results["metadatas"] and len(results["metadatas"][0]) > 0:
                    st.info(f"Đang tìm kiếm trong kho dữ liệu với từ khóa tối ưu: '{prompt}'")
                    paths = [meta["path"] for meta in results["metadatas"][0]]
                    distances = results["distances"][0]
                    
                    cols = st.columns(len(paths))
                    for col, path, dist in zip(cols, paths, distances):
                        # Distance càng nhỏ càng giống (vì dùng Cosine Distance)
                        confidence = max(0, 100 - (dist * 100))
                        col.image(Image.open(path), caption=f"{path.split('/')[-1]}\nĐộ khớp: {confidence:.1f}%")
                    
                    if min(distances) > 0.75:
                        st.warning("⚠️ Độ khớp khá thấp. Có thể trong tập dữ liệu (chỉ có ~59 ảnh) không có ảnh nào thực sự chứa nội dung bạn đang tìm.")
                else:
                    st.warning("Không tìm thấy kết quả. Vui lòng chạy index_data.py trước!")
        else:
            st.warning("Vui lòng nhập text!")

with tab2:
    st.header("Tìm ảnh bằng Hình ảnh")
    uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Query Image", width=300)
        
        if st.button("Tìm kiếm Image"):
            with st.spinner("Đang tìm kiếm..."):
                emb = extractor.get_image_embedding(image=image)
                results = db.search(emb, n_results=5)
                
                if results and results["metadatas"] and len(results["metadatas"][0]) > 0:
                    paths = [meta["path"] for meta in results["metadatas"][0]]
                    distances = results["distances"][0]
                    
                    cols = st.columns(len(paths))
                    for col, path, dist in zip(cols, paths, distances):
                        confidence = max(0, 100 - (dist * 100))
                        col.image(Image.open(path), caption=f"{path.split('/')[-1]}\nĐộ khớp: {confidence:.1f}%")
                else:
                    st.warning("Không tìm thấy kết quả. Vui lòng chạy index_data.py trước!")
