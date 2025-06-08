import os
from sklearn.datasets import fetch_20newsgroups

# Load the full dataset (you can also specify 'train' or 'test' subsets)
print("Initing newsgroups")
newsgroups = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

print("making dir?")
# Create a directory to store the documents
output_dir = "documents"
os.makedirs(output_dir, exist_ok=True)

# Save each document as a separate .txt file
for i, text in enumerate(newsgroups.data):
    print(f"doc{i} download")
    category = newsgroups.target_names[newsgroups.target[i]]
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    file_path = os.path.join(category_dir, f"doc_{i}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print(f"Saved {len(newsgroups.data)} documents to '{output_dir}' by category.")
