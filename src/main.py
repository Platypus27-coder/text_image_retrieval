import os
import sys

# Add project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.similarity import get_cosine_scores
from src.utils.visualization import plot_results

def main():
    ROOT = 'data'
    root_img_path = f"{ROOT}/train/"
    
    # Extract CLASS_NAME from train directory
    class_names = sorted(list(os.listdir(root_img_path)))
    
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    size = (400, 400)
    
    print(f"Start searching with query image: {query_path}")
    print("Extracting features and calculating Cosine Similarity...")
    query, ls_path_scores = get_cosine_scores(root_img_path, query_path, size, class_names)
    
    print("Displaying top 5 most similar images...")
    plot_results(query_path, ls_path_scores, reverse=True)

if __name__ == "__main__":
    main()
