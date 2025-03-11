import chromadb
import os

# Set your database path (Make sure it's correct)
DB_PATH = "test_image_embeddings"

# Initialize ChromaDB client
print("Initializing ChromaDB Client...")
if not os.path.exists(DB_PATH):
    print(f"Database path does not exist: {DB_PATH}")
else:
    print(f"Database path found: {DB_PATH}")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("image_embeddings")

# Check if database files exist
print("\n🔍 Checking database files...")
db_files = os.listdir(DB_PATH)
if not db_files:
    print("No files found in the database path!")

# Check the number of stored entries
entry_count = collection.count()
print(f"\n🔍 Number of stored entries in the database: {entry_count}")

if entry_count == 0:
    print("No embeddings found. Check if embeddings were added correctly.")

# Fetch a test entry
print("\n🔍 Fetching a test entry...")
test_entry = collection.get(limit=1)

if not test_entry or not test_entry.get("ids", []):
    print("No entries found in ChromaDB! Embeddings might not have been stored properly.")
else:
    print("Sample Entry Found:", test_entry)

# Fetch all embeddings
print("\n🔍 Retrieving all stored embeddings...")
all_embeddings = collection.get()

image_ids = all_embeddings.get('ids', [])
image_embeddings = all_embeddings.get('embeddings', [])
metadata_list = all_embeddings.get('metadatas', [])

if not image_embeddings:
    print("No embeddings found in the database.")
else:
    print(f"Found {len(image_embeddings)} embeddings in the database.")

# Print a sample embedding
if image_embeddings:
    print("\n🧐 Sample Stored Data:")
    print(f"🆔 ID: {image_ids[0]}")
    print(f"📊 Embedding: {image_embeddings[0][:10]}... (truncated for readability)")
    print(f"📁 Metadata: {metadata_list[0]}")

print("\n🎯 Debugging Completed.")
