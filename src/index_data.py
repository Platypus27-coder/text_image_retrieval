import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.clip_model import CLIPFeatureExtractor
from src.utils.vector_db import VectorDB

def main():
    extractor = CLIPFeatureExtractor()
    db = VectorDB()
    
    root_img_path = 'data/train/'
    class_names = sorted(list(os.listdir(root_img_path)))
    
    ids = []
    embeddings = []
    metadatas = []
    
    print("Bắt đầu indexing dữ liệu ảnh vào Vector DB...")
    
    for folder in class_names:
        folder_path = os.path.join(root_img_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        images = os.listdir(folder_path)
        print(f"Processing {folder} ({len(images)} images)...")
        
        for img_name in tqdm(images[:100]): # Tạm thời lấy 100 ảnh mỗi class để test cho nhanh
            img_path = os.path.join(folder_path, img_name)
            try:
                emb = extractor.get_image_embedding(image_path=img_path)
                ids.append(img_path)
                embeddings.append(emb)
                metadatas.append({"path": img_path, "class": folder})
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
            # Batch add
            if len(ids) >= 100:
                db.add_images(ids=ids, embeddings=embeddings, metadatas=metadatas)
                ids = []
                embeddings = []
                metadatas = []
                
    if len(ids) > 0:
        db.add_images(ids=ids, embeddings=embeddings, metadatas=metadatas)
        
    print("Hoàn tất Indexing! Dữ liệu đã lưu vào .chroma_data/")

if __name__ == "__main__":
    main()
