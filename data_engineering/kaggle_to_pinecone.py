import pandas as pd
import pinecone
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
import json

from pinecone import Pinecone, ServerlessSpec
load_dotenv()

# Kaggle and Pinecone setup
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
INDEX_NAME = "climate-change-opinions"
DATASET_NAME = "asaniczka/public-opinion-on-climate-change-updated-daily"
FILE_NAME = "reddit_opinion_climate_change.csv"

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Set API Key")

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    print("Index not found")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    print("Created index")

# Connect to the index
index = pc.Index(INDEX_NAME)
print("Set index")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def download_latest_data():
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            api.dataset_download_file(DATASET_NAME, FILE_NAME, path=tmpdir)
            df = pd.read_csv(os.path.join(tmpdir, FILE_NAME)).dropna()
            print("Downloaded latest data!")
        return df
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        print("Please check the following:")
        print(f"1. Does the dataset '{DATASET_NAME}' exist on Kaggle?")
        print(f"2. Is the file name '{FILE_NAME}' correct?")
        print("3. Do you have the necessary permissions to access this dataset?")
        print("4. Are your Kaggle API credentials correct?")
        
        # Fallback to local file if available
        local_file_path = '../data/reddit_opinion_climate_change.csv'
        if os.path.exists(local_file_path):
            print(f"Attempting to load data from local file: {local_file_path}")
            return pd.read_csv(local_file_path).dropna()
        else:
            print("Local file not found. Unable to proceed.")
            return None


def get_last_updated_time():
    # Query the last entry in Pinecone to get the last updated time
    # This is a placeholder - you'll need to implement this based on your data structure
    response = index.query(vector=[0]*384, top_k=1, include_metadata=True)
    if response['matches']:
        return pd.to_datetime(response['matches'][0]['metadata']['created_time'])
    return pd.Timestamp.min

def generate_embedding(text):
    return model.encode(text).tolist()

def truncate_text(text, max_length=1000):
    return text[:max_length] if text else ""

def process_and_upsert_data(df, last_updated_time):
    new_data = df[pd.to_datetime(df['created_time']) > last_updated_time]
    print(f"New data to process: {len(new_data)} rows")

    batch_size = 100  # Reduced from 800 to 100
    for i in range(0, len(new_data), batch_size):
        batch = new_data.iloc[i:i+batch_size]
        
        vectors = []
        for _, row in batch.iterrows():
            text_to_embed = f"{truncate_text(row['self_text'])} {truncate_text(row['post_title'])}"
            embedding = generate_embedding(text_to_embed)
            
            metadata = {
                "comment_id": row['comment_id'],
                "score": int(row['score']),
                "self_text": truncate_text(row['self_text']),
                "subreddit": row['subreddit'],
                "created_time": row['created_time'],
                "post_id": row['post_id'],
                "author_name": row['author_name'],
                "controversiality": int(row['controversiality']),
                "ups": int(row['ups']),
                "downs": int(row['downs']),
                "user_is_verified": bool(row['user_is_verified']),
                "user_account_created_time": row['user_account_created_time'],
                "user_awardee_karma": float(row['user_awardee_karma']),
                "user_awarder_karma": float(row['user_awarder_karma']),
                "user_link_karma": float(row['user_link_karma']),
                "user_comment_karma": float(row['user_comment_karma']),
                "user_total_karma": float(row['user_total_karma']),
                "post_score": int(row['post_score']),
                "post_self_text": truncate_text(row['post_self_text']),
                "post_title": truncate_text(row['post_title']),
                "post_upvote_ratio": float(row['post_upvote_ratio']),
                "post_thumbs_ups": int(row['post_thumbs_ups']),
                "post_total_awards_received": int(row['post_total_awards_received']),
                "post_created_time": row['post_created_time']
            }
            
            # Ensure metadata size is within limits
            while len(json.dumps(metadata)) > 40960:  # 40 KB limit for metadata
                # Truncate the longest text field
                longest_field = max(metadata, key=lambda k: len(str(metadata[k])) if isinstance(metadata[k], str) else 0)
                if isinstance(metadata[longest_field], str):
                    metadata[longest_field] = metadata[longest_field][:len(metadata[longest_field])//2]
                else:
                    # If we've truncated all text and still too big, we need to skip this record
                    print(f"Skipping record {row['comment_id']} due to oversized metadata")
                    break
            else:
                vectors.append((row['comment_id'], embedding, metadata))
        
        try:
            index.upsert(vectors=vectors)
            print(f"Upserted batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
            # You might want to implement retry logic here

def main():
    print("Starting data update process...")
    last_updated_time = get_last_updated_time()
    print(f"Last updated time: {last_updated_time}")

    df = download_latest_data()
    print(f"Downloaded data shape: {df.shape}")

    process_and_upsert_data(df, last_updated_time)
    print("Data update complete!")

if __name__ == "__main__":
    main()