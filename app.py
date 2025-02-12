# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import requests
# import os
# from dotenv import load_dotenv

# app = Flask(__name__)
# # Allow requests from http://localhost:3000
# # CORS(app, resources={r"/transcribe": {"origins": "https://speech-to-text-frontend-1ryq.vercel.app"}})
# CORS(app)  # Temporary fix to allow all origins


# # Hugging Face API details
# WAV2VEC_API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
# BERT_API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
# T5_API_URL = "https://api-inference.huggingface.co/models/t5-small"

# load_dotenv()  # Load variables from .env file
# HF_API_TOKEN  = os.getenv("HF_API_KEY")

# # Headers for Hugging Face API requests
# headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# def transcribe_audio(audio_data):
#     """Transcribe audio using Wav2Vec 2.0 via Hugging Face API."""
#     response = requests.post(WAV2VEC_API_URL, headers=headers, data=audio_data)
#     if response.status_code == 200:
#         return response.json()["text"]
#     return None

# def refine_with_bert(text):
#     """Refine transcription using BERT via Hugging Face API."""
#     payload = {"inputs": text}
#     response = requests.post(BERT_API_URL, headers=headers, json=payload)
#     if response.status_code == 200:
#         return response.json()[0]["generated_text"]
#     return text  # Fallback to original text if API fails

# def refine_with_t5(text):
#     """Refine transcription using T5 via Hugging Face API."""
#     payload = {"inputs": f"correct: {text}"}
#     response = requests.post(T5_API_URL, headers=headers, json=payload)
#     if response.status_code == 200:
#         # Inspect the response structure
#         response_data = response.json()
#         if isinstance(response_data, list) and len(response_data) > 0:
#             # Assuming the response is a list of dictionaries with a "generated_text" key
#             return response_data[0].get("generated_text", text)
#         elif isinstance(response_data, dict):
#             # If the response is a dictionary, look for the correct key
#             return response_data.get("generated_text", text)
#         else:
#             # Fallback to original text if the response structure is unexpected
#             return text
#     return text  # Fallback to original text if API fails

# @app.route("/transcribe", methods=["POST"])
# def transcribe():
#     """Endpoint to handle audio transcription and refinement."""
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     audio_file = request.files["file"]
#     try:
#         audio_data = audio_file.read()
#         if not audio_data:
#             return jsonify({"error": "Uploaded file is empty"}), 400

#         # Step 1: Transcribe audio using Wav2Vec 2.0
#         transcription = transcribe_audio(audio_data)
#         if not transcription:
#             return jsonify({"error": "Failed to transcribe audio"}), 500

#         # Step 2: Refine transcription using BERT
#         bert_refined = refine_with_bert(transcription)

#         # Step 3: Further refine transcription using T5
#         final_transcription = refine_with_t5(bert_refined)

#         return jsonify({"transcript": final_transcription})
#     except Exception as e:
#         print(f"Error processing audio: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv

app = Flask(__name__)
# Allow requests from http://localhost:3000
# CORS(app, resources={r"/transcribe": {"origins": "https://speech-to-text-frontend-1ryq.vercel.app"}})
# Allow all origins for now (you can restrict this to your frontend URL later)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

REVIEWS_FILE = "reviews.json"

# Hugging Face API details
WAV2VEC_API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
BERT_API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"
T5_API_URL = "https://api-inference.huggingface.co/models/t5-small"

load_dotenv()  # Load variables from .env file
HF_API_TOKEN  = os.getenv("HF_API_KEY")

# Headers for Hugging Face API requests
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def transcribe_audio(audio_data):
    """Transcribe audio using Wav2Vec 2.0 via Hugging Face API."""
    response = requests.post(WAV2VEC_API_URL, headers=headers, data=audio_data)
    if response.status_code == 200:
        return response.json()["text"]
    return None

def refine_with_bert(text):
    """Refine transcription using BERT via Hugging Face API."""
    payload = {"inputs": text}
    response = requests.post(BERT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    return text  # Fallback to original text if API fails

def refine_with_t5(text):
    """Refine transcription using T5 via Hugging Face API."""
    payload = {"inputs": f"correct: {text}"}
    response = requests.post(T5_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        # Inspect the response structure
        response_data = response.json()
        if isinstance(response_data, list) and len(response_data) > 0:
            # Assuming the response is a list of dictionaries with a "generated_text" key
            return response_data[0].get("generated_text", text)
        elif isinstance(response_data, dict):
            # If the response is a dictionary, look for the correct key
            return response_data.get("generated_text", text)
        else:
            # Fallback to original text if the response structure is unexpected
            return text
    return text  # Fallback to original text if API fails

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Endpoint to handle audio transcription and refinement."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]
    try:
        audio_data = audio_file.read()
        if not audio_data:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # Step 1: Transcribe audio using Wav2Vec 2.0
        transcription = transcribe_audio(audio_data)
        if not transcription:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        # Step 2: Refine transcription using BERT
        bert_refined = refine_with_bert(transcription)

        # Step 3: Further refine transcription using T5
        final_transcription = refine_with_t5(bert_refined)

        return jsonify({"transcript": final_transcription})
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Ensure the reviews file exists
if not os.path.exists(REVIEWS_FILE):
    with open(REVIEWS_FILE, "w") as file:
        json.dump([], file)

def load_reviews():
    """Load reviews from JSON file."""
    try:
        with open(REVIEWS_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []

# def save_review(review):
#     """Save a new review to the JSON file."""
#     reviews = load_reviews()
#     reviews.append(review)
#     with open(REVIEWS_FILE, "w") as file:
#         json.dump(reviews, file, indent=4)

def save_reviews(reviews):
    """Save only valid reviews to the JSON file, preventing empty lists."""
    cleaned_reviews = [review for review in reviews if isinstance(review, dict) and review]  # Remove empty lists/dictionaries

    with open(REVIEWS_FILE, "w") as file:
        json.dump(cleaned_reviews, file, indent=4)


@app.route("/submit-review", methods=["POST"])
def submit_review():
    try:
        data = request.get_json()
        print("Received Data:", data)  # 🔍 Debugging line

        rating = data.get("rating")
        feedback = data.get("feedback")
        transcript = data.get("transcript", "")  # Ensure transcript is always included

        if not rating or not feedback:
            return jsonify({"error": "Missing required fields"}), 400  # Ensure required fields exist

        # Load existing reviews
        reviews = []
        file_path = "reviews.json"
        try:
            with open(file_path, "r") as file:
                reviews = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Add new review
        new_review = {
            "rating": rating,
            "feedback": feedback,
            "transcript": transcript  # Ensure transcript is always included
        }
        reviews.append(new_review)

        # Save back to JSON
        with open(file_path, "w") as file:
            json.dump(reviews, file, indent=4)

        return jsonify({"message": "Review submitted successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get-reviews", methods=["GET"])
def get_reviews():
    """Endpoint to retrieve all reviews."""
    return jsonify(load_reviews())

@app.route("/delete-review/<int:index>", methods=["DELETE"])
def delete_review(index):
    """Delete a review permanently from the JSON file."""
    reviews = load_reviews()

    if 0 <= index < len(reviews):  
        del reviews[index]  # Remove the review at the given index
        save_reviews(reviews)  # Save the updated list
        return jsonify({"message": "Review deleted successfully"}), 200
    else:
        return jsonify({"error": "Invalid review index"}), 400


if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0", port=5000)