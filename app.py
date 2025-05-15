# app.py
import os
import sys # For flushing print statements
import logging
import urllib.error
import traceback # Added for detailed exception logging
from urllib.parse import urlparse, parse_qs # For robust URL parsing

# Explicitly load .env file AT THE VERY TOP (for local development)
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("DEBUG: .env file loaded successfully by load_dotenv().", flush=True)
    else:
        print("DEBUG: .env file not found by load_dotenv() or it's empty. Relying on system env vars.", flush=True)
except ImportError:
    print("DEBUG: python-dotenv library is not installed. .env file will not be loaded by load_dotenv().", flush=True)

# Configure basic logging ASAP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - L%(lineno)d - %(message)s')
logger = logging.getLogger(__name__) # Use this logger for app-specific messages

# Print raw environment variables immediately after attempting to load .env using the logger
logger.info("--- RAW ENVIRONMENT VARIABLE VALUES (immediately after dotenv attempt) ---")
logger.info(f"RAW FLASK_SECRET_KEY: '{os.environ.get('FLASK_SECRET_KEY')}'")
logger.info(f"RAW TRANSLATOR_ENDPOINT: '{os.environ.get('TRANSLATOR_ENDPOINT')}'")
logger.info(f"RAW TRANSLATOR_SUBSCRIPTION_KEY: '{os.environ.get('TRANSLATOR_SUBSCRIPTION_KEY')}'")
logger.info(f"RAW TRANSLATOR_REGION: '{os.environ.get('TRANSLATOR_REGION')}'")
logger.info(f"RAW LANGUAGE_ENDPOINT: '{os.environ.get('LANGUAGE_ENDPOINT')}'")
logger.info(f"RAW LANGUAGE_SUBSCRIPTION_KEY: '{os.environ.get('LANGUAGE_SUBSCRIPTION_KEY')}'")
logger.info(f"RAW SPEECH_ENDPOINT (not typically used directly by SDK, region is key): '{os.environ.get('SPEECH_ENDPOINT')}'")
logger.info(f"RAW SPEECH_SUBSCRIPTION_KEY: '{os.environ.get('SPEECH_SUBSCRIPTION_KEY')}'")
logger.info(f"RAW SPEECH_REGION: '{os.environ.get('SPEECH_REGION')}'")
logger.info(f"RAW VISION_ENDPOINT: '{os.environ.get('VISION_ENDPOINT')}'")
logger.info(f"RAW VISION_SUBSCRIPTION_KEY: '{os.environ.get('VISION_SUBSCRIPTION_KEY')}'")
logger.info("--- END RAW ENVIRONMENT VALUES ---")


import requests
import PyPDF2
import math
import time
import json

from io import BytesIO

from flask import (Flask, render_template, request, jsonify, Response,
                   send_from_directory, redirect, url_for, flash, session)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from pytube import YouTube
from pytube.exceptions import PytubeError, VideoUnavailable, VideoPrivate, RegexMatchError
# Import for new YouTube transcript method
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled


# --- Azure SDK Imports ---
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient, ExtractiveSummaryAction, AbstractiveSummaryAction
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem, TranslatedTextItem, Translation # Added Translation
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ServiceRequestError

# --- Database & Login Imports ---
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
import nltk

# --- PDF Export ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# --- Initialize Flask App ---
app = Flask(__name__)
app_logger = app.logger
app_logger.info("Flask app object created.")

# --- Configuration ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-dev-secret-key-SHOULD-BE-SET-IN-ENV')
if app.config['SECRET_KEY'] == 'default-dev-secret-key-SHOULD-BE-SET-IN-ENV':
    app_logger.warning("CRITICAL: FLASK_SECRET_KEY is using the default development value. SET THIS IN YOUR .env FILE or App Service Configuration!")

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, os.environ.get('UPLOAD_FOLDER_NAME', 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db_uri = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.join(app.root_path, 'instance', 'toolkit.db')}")
if db_uri and db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Azure Service Credentials ---
# These will be read from environment variables set by Azure App Service or Docker
TRANSLATOR_KEY = os.environ.get('TRANSLATOR_SUBSCRIPTION_KEY')
TRANSLATOR_ENDPOINT_URL = os.environ.get('TRANSLATOR_ENDPOINT')
TRANSLATOR_REGION = os.environ.get('TRANSLATOR_REGION')

LANGUAGE_KEY = os.environ.get('LANGUAGE_SUBSCRIPTION_KEY')
LANGUAGE_ENDPOINT_URL = os.environ.get('LANGUAGE_ENDPOINT')

SPEECH_KEY = os.environ.get('SPEECH_SUBSCRIPTION_KEY')
SPEECH_REGION = os.environ.get('SPEECH_REGION')

VISION_KEY = os.environ.get('VISION_SUBSCRIPTION_KEY')
VISION_ENDPOINT_URL = os.environ.get('VISION_ENDPOINT')

# --- Initialize Azure Clients ---
app_logger.info("Attempting to initialize Azure clients with parsed env values...")

def is_valid_endpoint_url(url_string, service_name="Service"):
    if not url_string:
        app_logger.error(f"{service_name} endpoint URL is missing or empty from environment variables.")
        return False
    cleaned_url = str(url_string).strip().strip("'").strip('"')
    if not cleaned_url.startswith("https://"):
        app_logger.error(f"Invalid {service_name} Endpoint URL: '{cleaned_url}'. It must start with 'https://'.")
        return False
    if "#" in cleaned_url[8:] or "\n" in cleaned_url or "\r" in cleaned_url:
        app_logger.error(f"Invalid characters (like # or newlines) found in {service_name} Endpoint URL: '{cleaned_url}'.")
        return False
    return True

translator_client = None
if all([TRANSLATOR_KEY, TRANSLATOR_ENDPOINT_URL, TRANSLATOR_REGION]):
    try:
        cleaned_translator_endpoint = str(TRANSLATOR_ENDPOINT_URL).strip().strip("'").strip('"')
        if not is_valid_endpoint_url(cleaned_translator_endpoint, "Translator"):
            raise ValueError("Invalid Translator Endpoint URL format or content after cleaning.")
        translator_credential = TranslatorCredential(TRANSLATOR_KEY, TRANSLATOR_REGION)
        translator_client = TextTranslationClient(endpoint=cleaned_translator_endpoint, credential=translator_credential)
        app_logger.info(f"Translator client initialized successfully with endpoint: {cleaned_translator_endpoint}")
    except ValueError as ve:
        app_logger.error(f"ValueError during Translator client initialization: {ve}")
    except Exception as e:
        app_logger.error(f"Failed to initialize Translator client with raw endpoint '{TRANSLATOR_ENDPOINT_URL}': {e}\n{traceback.format_exc()}")
else:
    app_logger.warning("Translator service credentials not fully configured. Check TRANSLATOR_SUBSCRIPTION_KEY, TRANSLATOR_ENDPOINT, TRANSLATOR_REGION.")

text_analytics_client = None
if all([LANGUAGE_KEY, LANGUAGE_ENDPOINT_URL]):
    try:
        cleaned_language_endpoint = str(LANGUAGE_ENDPOINT_URL).strip().strip("'").strip('"')
        if not is_valid_endpoint_url(cleaned_language_endpoint, "Language"):
            raise ValueError("Invalid Language Endpoint URL format or content after cleaning.")
        text_analytics_client = TextAnalyticsClient(endpoint=cleaned_language_endpoint, credential=AzureKeyCredential(LANGUAGE_KEY))
        app_logger.info(f"Text Analytics client initialized successfully with endpoint: {cleaned_language_endpoint}")
    except ValueError as ve:
        app_logger.error(f"ValueError during Text Analytics client initialization: {ve}")
    except Exception as e:
        app_logger.error(f"Failed to initialize Text Analytics client with raw endpoint '{LANGUAGE_ENDPOINT_URL}': {e}\n{traceback.format_exc()}")
else:
    app_logger.warning("Language service (Text Analytics) credentials not fully configured. Check LANGUAGE_SUBSCRIPTION_KEY, LANGUAGE_ENDPOINT.")

speech_config = None
if all([SPEECH_KEY, SPEECH_REGION]):
    try:
        app_logger.info(f"Initializing Speech client with Region: '{SPEECH_REGION}' and Key ending: '...{SPEECH_KEY[-4:] if SPEECH_KEY and len(SPEECH_KEY) >=4 else 'N/A'}'")
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_recognition_language="en-US"
        app_logger.info("Speech config initialized successfully.")
    except Exception as e:
        app_logger.error(f"Failed to initialize Speech config: {e}\n{traceback.format_exc()}")
else:
    app_logger.warning("Speech service credentials not fully configured. Check SPEECH_SUBSCRIPTION_KEY, SPEECH_REGION.")

computervision_client = None
if all([VISION_KEY, VISION_ENDPOINT_URL]):
    try:
        cleaned_vision_endpoint = str(VISION_ENDPOINT_URL).strip().strip("'").strip('"')
        app_logger.info(f"Attempting to initialize Computer Vision client with cleaned endpoint: '{cleaned_vision_endpoint}'")
        if not is_valid_endpoint_url(cleaned_vision_endpoint, "Vision"):
            raise ValueError("Invalid Vision Endpoint URL format or content after cleaning.")
        computervision_client = ComputerVisionClient(cleaned_vision_endpoint, CognitiveServicesCredentials(VISION_KEY))
        app_logger.info(f"Computer Vision client initialized successfully with endpoint: {cleaned_vision_endpoint}")
    except ValueError as ve:
        app_logger.error(f"ValueError during Computer Vision client initialization: {ve}")
    except Exception as e:
        app_logger.error(f"Failed to initialize Computer Vision client with raw endpoint '{VISION_ENDPOINT_URL}': {e}\n{traceback.format_exc()}")
else:
    app_logger.warning("Vision service credentials not fully configured. Check VISION_SUBSCRIPTION_KEY, VISION_ENDPOINT.")


# --- Database Setup ---
db = SQLAlchemy(app)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    summaries = db.relationship('Summary', backref='author', lazy=True)

class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=True)
    summarized_text = db.Column(db.Text, nullable=False)
    input_type = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    language = db.Column(db.String(10), nullable=True)

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- NLTK Setup (MODIFIED FOR DOCKER) ---
# Use the NLTK_DATA environment variable if set (e.g., by Dockerfile),
# otherwise default to a local path (relative to app.root_path for flexibility).
# In Docker, NLTK_DATA will be /app/nltk_data_local as set in Dockerfile.
nltk_data_dir = os.environ.get('NLTK_DATA', os.path.join(app.root_path, 'nltk_data_local'))

if not os.path.exists(nltk_data_dir):
    try:
        os.makedirs(nltk_data_dir)
        app_logger.info(f"Created NLTK data directory: {nltk_data_dir}")
    except OSError as e:
        app_logger.error(f"Could not create NLTK data directory {nltk_data_dir}: {e}")

# Ensure this path is always in nltk.data.path for downloads and lookups
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)
app_logger.info(f"NLTK data path configured: {nltk.data.path}")

def ensure_nltk_punkt():
    try:
        # Check if 'punkt' can be found in any of the NLTK data paths
        # Using .zip is a more reliable way to check for the punkt resource
        nltk.data.find('tokenizers/punkt.zip')
        app_logger.info(f"'punkt' tokenizer found by NLTK in paths: {nltk.data.path}.")
    except LookupError:
        app_logger.info(f"'punkt' not found. Attempting download to specified NLTK_DATA directory: {nltk_data_dir}...")
        try:
            # Explicitly set download_dir to the path configured (either from ENV or default)
            nltk.download('punkt', download_dir=nltk_data_dir)
            app_logger.info(f"NLTK 'punkt' downloaded successfully to {nltk_data_dir}")
        except Exception as e:
            app_logger.error(f"Failed to download NLTK 'punkt' to {nltk_data_dir}: {e}. Summarization features might be affected.")
            # Optionally, re-raise or handle if 'punkt' is absolutely critical for app startup
ensure_nltk_punkt()
# --- END OF NLTK Setup (MODIFIED FOR DOCKER) ---

# --- Helper Functions ---
ALLOWED_EXTENSIONS_PDF = {'pdf'}
ALLOWED_EXTENSIONS_AUDIO = {'wav', 'mp3', 'ogg', 'm4a', 'mp4'}
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_pdf_text_pypdf2(pdf_file_stream):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text if text else None
    except Exception as e:
        app_logger.error(f"PyPDF2 PDF extraction failed: {e}\n{traceback.format_exc()}")
        return None

def get_youtube_video_id(url):
    """Extracts YouTube video ID from various URL formats."""
    if not url:
        return None
    try:
        # Standard YouTube URLs
        if "youtube.com/watch?v=" in url: # Example, adjust if your actual URLs are different
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url: # Example
            return url.split("youtu.be/")[1].split("?")[0]

        parsed_url = urlparse(url)
        if "googleusercontent.com/youtube.com" in parsed_url.netloc:
            if parsed_url.path == "/watch":
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get("v", [None])[0]
                if video_id and len(video_id) == 11: # Standard YouTube ID length
                    app_logger.info(f"Extracted video ID '{video_id}' from URL '{url}' using 'v' parameter.")
                    return video_id
            else:
                path_parts = parsed_url.path.strip('/').split('/')
                if path_parts:
                    potential_id = path_parts[-1]
                    if len(potential_id) == 11:
                         app_logger.info(f"Extracted potential video ID '{potential_id}' from URL path '{url}'.")
                         return potential_id
    except Exception as e:
        app_logger.error(f"Error parsing YouTube URL '{url}' to get video ID: {e}")

    app_logger.warning(f"Could not extract a valid video ID from YouTube URL: {url}")
    return None


def get_transcript_from_youtube_api(video_url):
    """Attempts to get a transcript using youtube_transcript_api."""
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return None, None, "Could not extract Video ID from URL for Transcript API."

    video_title = f"YouTube Video ({video_id})" # Default title
    try:
        # Attempt to fetch title using Pytube, but don't let it block transcript fetching
        try:
            yt_temp = YouTube(f"https://www.youtube.com/watch?v=VIDEO_ID{video_id}") # Construct a standard URL for Pytube
            video_title = yt_temp.title
            app_logger.info(f"Fetched title '{video_title}' using Pytube for transcript API processing.")
        except Exception as title_e:
            app_logger.warning(f"Could not fetch YouTube title with Pytube for {video_url} (ID: {video_id}) (transcript API context): {title_e}")


        app_logger.info(f"Attempting to fetch transcript for video ID: {video_id} (Title: {video_title}) using youtube_transcript_api.")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript_obj = None # Use a more descriptive name
        preferred_languages = ['en', 'en-US', 'en-GB']
        # Try manually created transcripts first
        for lang_code in preferred_languages:
            try:
                transcript_obj = transcript_list.find_manually_created_transcript([lang_code])
                app_logger.info(f"Found manual transcript in '{lang_code}' for {video_id}.")
                break
            except NoTranscriptFound:
                continue

        # If no manual transcript, try generated ones
        if not transcript_obj:
            for lang_code in preferred_languages:
                try:
                    transcript_obj = transcript_list.find_generated_transcript([lang_code])
                    app_logger.info(f"Found auto-generated transcript in '{lang_code}' for {video_id}.")
                    break
                except NoTranscriptFound:
                    continue

        # If still no preferred language transcript, try any available one
        if not transcript_obj:
            app_logger.info(f"No preferred language transcript found for {video_id}. Trying any available transcript.")
            for available_transcript in transcript_list: # This iterates over Transcript objects
                transcript_obj = available_transcript
                app_logger.info(f"Found an available transcript in language: {transcript_obj.language} (code: {transcript_obj.language_code}) for {video_id}")
                break

        if transcript_obj:
            transcript_data = transcript_obj.fetch()
            transcript_text = " ".join([entry['text'] for entry in transcript_data])
            app_logger.info(f"Successfully fetched transcript for video ID: {video_id} using youtube_transcript_api.")
            return transcript_text, video_title, None
        else:
            app_logger.warning(f"No transcript (manual, auto, or any other) could be loaded for video ID: {video_id}")
            return None, video_title, "No transcript could be loaded for this video via API."

    except TranscriptsDisabled:
        app_logger.warning(f"Transcripts are disabled for video ID: {video_id}.")
        return None, video_title, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        app_logger.warning(f"No transcript found for video ID: {video_id} (API direct error).")
        return None, video_title, "No transcript found for this video via API."
    except Exception as e:
        app_logger.error(f"Error using youtube_transcript_api for video ID {video_id}: {type(e).__name__} - {e}\n{traceback.format_exc()}")
        return None, video_title, f"API error fetching transcript: {type(e).__name__}."


def get_youtube_audio_stream_pytube(video_url):
    """Attempts to get an audio stream from a YouTube URL using Pytube."""
    if not video_url:
        app_logger.error("No YouTube URL provided to Pytube.")
        return None, None, "YouTube URL cannot be empty."
    try:
        app_logger.info(f"Attempting to process YouTube URL with Pytube: {video_url}")
        yt = YouTube(video_url)
        video_title = yt.title
        app_logger.info(f"Pytube object created for URL: {video_url}. Video Title: '{video_title}'")

        app_logger.info(f"Pytube: Checking availability for video: '{video_title}'...")
        yt.check_availability() # This can raise VideoUnavailable or VideoPrivate
        app_logger.info(f"Pytube: Video '{video_title}' passed availability check.")

        app_logger.info(f"Pytube: Filtering for audio streams for video: '{video_title}'...")
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').desc().first()
        if not audio_stream:
            app_logger.info(f"Pytube: No mp4 audio stream found for '{video_title}', trying webm.")
            audio_stream = yt.streams.filter(only_audio=True, file_extension='webm').order_by('abr').desc().first()

        if audio_stream:
            app_logger.info(f"Pytube: Found audio stream for '{video_title}': {audio_stream}")
            audio_buffer = BytesIO()
            audio_stream.stream_to_buffer(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer, video_title, None
        else:
            app_logger.warning(f"Pytube: No suitable audio stream (mp4 or webm) found for YouTube URL: {video_url} (Title: '{video_title}')")
            return None, video_title, f"Pytube: No suitable audio stream found for '{video_title}' (tried mp4 and webm)."

    except VideoUnavailable:
        app_logger.error(f"Pytube error: Video {video_url} is unavailable.")
        title_on_error = f"YouTube Video ({video_url} - Unavailable)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        return None, title_on_error, "Pytube: This YouTube video is unavailable."
    except VideoPrivate:
        app_logger.error(f"Pytube error: Video {video_url} is private.")
        title_on_error = f"YouTube Video ({video_url} - Private)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        return None, title_on_error, "Pytube: This YouTube video is private."
    except RegexMatchError as rme:
        app_logger.error(f"Pytube RegexMatchError for URL '{video_url}': {rme}")
        title_on_error = f"YouTube Video ({video_url} - Regex Error)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        return None, title_on_error, "Pytube could not process this video due to internal regex matching issues."
    except PytubeError as pe:
        app_logger.error(f"Pytube specific error for URL '{video_url}': {type(pe).__name__} - {pe}")
        title_on_error = f"YouTube Video ({video_url} - Pytube Error)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        if "regex_search" in str(pe).lower() or "cipher" in str(pe).lower():
            return None, title_on_error, "Pytube could not process this video, possibly due to YouTube updates or video restrictions (Cipher/Regex error)."
        return None, title_on_error, f"Pytube: Could not process YouTube URL (PytubeError: {type(pe).__name__})."
    except urllib.error.HTTPError as httpe:
        app_logger.error(f"Pytube: HTTPError processing YouTube URL '{video_url}': {httpe.code} {httpe.reason}\n{traceback.format_exc()}")
        title_on_error = f"YouTube Video ({video_url} - HTTP Error)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        if httpe.code == 400:
             return None, title_on_error, "Pytube: Could not process YouTube URL: Bad Request (HTTP 400). Ensure the URL is a valid video page."
        return None, title_on_error, f"Pytube: HTTP error {httpe.code} processing YouTube URL: {httpe.reason}"
    except Exception as e:
        app_logger.error(f"Pytube: Generic unexpected error processing YouTube URL '{video_url}': {type(e).__name__} - {e}\n{traceback.format_exc()}")
        title_on_error = f"YouTube Video ({video_url} - Unexpected Error)"
        try: title_on_error = YouTube(video_url, use_oauth=False, allow_oauth_cache=False).title
        except: pass
        return None, title_on_error, f"Pytube: An unexpected error occurred ({type(e).__name__}). Check server logs for details."

# --- Azure Service Functions ---
# (translate_text_azure, analyze_text_sentiment_azure, summarize_text_azure,
#  transcribe_audio_azure, analyze_image_azure functions remain the same as your uploaded app.py)
# --- For brevity, I'm not repeating them here, but they should be included ---
def translate_text_azure(text_to_translate, target_languages=["fr", "es"], source_language=None):
    if not translator_client:
        app_logger.error("translate_text_azure called but translator_client is not initialized.")
        return {"error": "Translator service not available or not configured."}
    try:
        input_text_elements = [InputTextItem(text=text_to_translate)]
        
        api_params = {'to': target_languages}
        if source_language and source_language.strip():
            app_logger.info(f"Translating with source_language: {source_language}")
            api_params['from_parameter'] = source_language 
        else:
            app_logger.info("Translating without source_language (auto-detect).")
        
        response_items = translator_client.translate(content=input_text_elements, **api_params)
        
        translations = {}
        if not response_items:
            app_logger.info("Translation response from Azure was empty or None.")
            return {"translations": {}, "message": "Translation service returned no results."}

        for item in response_items: # item is TranslatedTextItem
            if not isinstance(item, TranslatedTextItem):
                app_logger.error(f"Unexpected item type '{type(item)}' in translation response. Expected TranslatedTextItem. Skipping item.")
                continue

            if item.translations: # This is a list of Translation objects
                for translation_obj in item.translations:
                    if isinstance(translation_obj, Translation) and hasattr(translation_obj, 'text') and translation_obj.text is not None and hasattr(translation_obj, 'to') and translation_obj.to:
                        translations[translation_obj.to] = translation_obj.text
                    else:
                        lang_to = translation_obj.to if hasattr(translation_obj, 'to') else 'unknown'
                        app_logger.warning(f"Translation object for language '{lang_to}' is missing text or target language. Object: {translation_obj}")
            else:
                app_logger.warning(f"TranslatedTextItem has no 'translations' list or it's empty. Item: {item}")
        
        if not translations and response_items: 
            app_logger.warning(f"Translation call succeeded but no translated text was extracted. Response items: {response_items}")
            return {"error": "Translation succeeded but no text was returned. The input might not be translatable or an issue occurred.", "details": str(response_items)}

        return {"translations": translations}

    except TypeError as te: 
        app_logger.error(f"Azure translation TypeError (likely SDK argument issue for translate()): {te}\n{traceback.format_exc()}")
        return {"error": f"Translator SDK call failed due to argument mismatch or type issue. Details: {str(te)}"}
    except ClientAuthenticationError as cae:
        app_logger.error(f"Azure translation authentication error: {cae}\n{traceback.format_exc()}")
        return {"error": "Translator authentication failed. Check your subscription key and region."}
    except ServiceRequestError as sre: 
        app_logger.error(f"Azure translation service request error: {sre}\n{traceback.format_exc()}")
        return {"error": f"Could not connect to Translator service: {sre.message}. Check endpoint and network."}
    except HttpResponseError as hre: 
        app_logger.error(f"Azure translation HTTP error: {hre}\n{traceback.format_exc()}")
        error_message = f"Translator service request failed: {hre.message} (Status: {hre.status_code})"
        if hre.error and hasattr(hre.error, 'message'): 
            error_message += f" Details: {hre.error.message}"
        elif hre.response and hasattr(hre.response, 'text') and hre.response.text:
             try:
                error_content = json.loads(hre.response.text)
                if 'error' in error_content and 'message' in error_content['error']:
                    error_message += f" Azure API Error: {error_content['error']['message']}"
             except json.JSONDecodeError:
                error_message += f" Raw Response: {hre.response.text[:200]}" 
        return {"error": error_message}
    except Exception as e: 
        app_logger.error(f"Unexpected Azure translation error: {e}\n{traceback.format_exc()}")
        return {"error": f"An unexpected error occurred during translation: {str(e)}"}

def analyze_text_sentiment_azure(text_to_analyze):
    if not text_analytics_client:
        app_logger.error("analyze_text_sentiment_azure called but text_analytics_client is not initialized.")
        return {"error": "Language service not available or not configured."}
    try:
        documents = [text_to_analyze]
        response_list = text_analytics_client.analyze_sentiment(documents=documents, show_opinion_mining=True)
        
        if not response_list:
            return {"error": "Sentiment analysis returned an empty response."}

        response_item = response_list[0] 

        if response_item.is_error:
            err = response_item.error
            app_logger.error(f"Sentiment analysis error: Code: {err.code}, Message: {err.message}")
            return {"error": f"Sentiment analysis error: Code: {err.code}, Message: {err.message}"}
        
        opinions_data = []
        for sentence in response_item.sentences: # sentence is a SentenceSentiment object
            for opinion in sentence.mined_opinions: # opinion is a MinedOpinion object
                opinions_data.append({
                    "target": opinion.target.text,
                    "sentiment": opinion.target.sentiment,
                    "assessments": [assessment.text for assessment in opinion.assessments]
                })

        return {
            "sentiment": response_item.sentiment,
            "confidence_scores": {
                "positive": response_item.confidence_scores.positive,
                "neutral": response_item.confidence_scores.neutral,
                "negative": response_item.confidence_scores.negative,
            },
            "sentences_sentiments": [{
                "text": sentence.text,
                "sentiment": sentence.sentiment,
                "length": sentence.length,      # CORRECTED: Was grapheme_length
                "offset": sentence.offset        # CORRECTED: Was grapheme_offset
            } for sentence in response_item.sentences],
            "mined_opinions": opinions_data
        }
    except ClientAuthenticationError as cae: 
        app_logger.error(f"Azure Language authentication error: {cae}\n{traceback.format_exc()}")
        return {"error": "Language service authentication failed. Check your subscription key and endpoint."}
    except ServiceRequestError as sre:
        app_logger.error(f"Azure Language service request error: {sre}\n{traceback.format_exc()}")
        return {"error": f"Could not connect to Language service: {sre.message}. Check endpoint and network."}
    except HttpResponseError as hre:
        app_logger.error(f"Azure Language HTTP error: {hre}\n{traceback.format_exc()}")
        error_details = f"Service request failed: {hre.message} (Status: {hre.status_code})."
        if hre.response and hasattr(hre.response, 'text') and hre.response.text:
            try:
                error_content = json.loads(hre.response.text)
                if 'error' in error_content and 'message' in error_content['error']:
                    error_details += f" Azure details: {error_content['error']['message']}"
            except json.JSONDecodeError: 
                error_details += f" Raw response snippet: {hre.response.text[:200]}" 
        return {"error": error_details}
    except Exception as e:
        app_logger.error(f"Azure sentiment analysis error: {e}\n{traceback.format_exc()}")
        return {"error": f"An unexpected error occurred during sentiment analysis: {str(e)}"}


def summarize_text_azure(text_to_summarize, kind="extractive", max_sentence_count=3):
    if not text_analytics_client:
        app_logger.error("summarize_text_azure called but text_analytics_client is not initialized.")
        return {"error": "Language service not available or not configured."}
    try:
        actions_to_perform = []
        if kind == "extractive":
            actions_to_perform.append(ExtractiveSummaryAction(max_sentence_count=max_sentence_count))
        elif kind == "abstractive":
            actions_to_perform.append(AbstractiveSummaryAction(sentence_count=int(max_sentence_count)))
        else:
            return {"error": "Invalid summarization kind specified."}

        poller = text_analytics_client.begin_analyze_actions(
            documents=[text_to_summarize],
            actions=actions_to_perform,
        )
        result_pages = poller.result() 
        summary_text = ""

        for page in result_pages: 
            for doc_result in page: 
                if hasattr(doc_result, 'error') and doc_result.error: 
                    err = doc_result.error
                    app_logger.error(f"Summarization action error for a document: Code: {err.code}, Message: {err.message}")
                    return {"error": f"Summarization action failed for the document: {err.message} (Code: {err.code})"}
                
                if kind == "extractive" and doc_result.kind == "ExtractiveSummarization":
                    summary_text = " ".join([sentence.text for sentence in doc_result.sentences])
                    break 
                elif kind == "abstractive" and doc_result.kind == "AbstractiveSummarization":
                    summary_text = " ".join([summary.text for summary in doc_result.summaries]) 
                    break 
            if summary_text: break 
        
        return {"summary": summary_text if summary_text else "No summary could be generated or action result was unexpected."}

    except ClientAuthenticationError as cae:
        app_logger.error(f"Azure Language (summarization) authentication error: {cae}\n{traceback.format_exc()}")
        return {"error": "Language service authentication failed for summarization. Check key/endpoint."}
    except ServiceRequestError as sre:
        app_logger.error(f"Azure Language (summarization) service request error: {sre}\n{traceback.format_exc()}")
        return {"error": f"Could not connect to Language service for summarization: {sre.message}. Check endpoint and network."}
    except HttpResponseError as hre:
        app_logger.error(f"Azure Language (summarization) HTTP error: {hre}\n{traceback.format_exc()}")
        error_details = f"Summarization service request failed: {hre.message} (Status: {hre.status_code})."
        if hre.response and hasattr(hre.response, 'text') and hre.response.text:
            try:
                error_content = json.loads(hre.response.text)
                if 'error' in error_content and 'message' in error_content['error']:
                    error_details += f" Azure details: {error_content['error']['message']}"
            except json.JSONDecodeError:
                 error_details += f" Raw response snippet: {hre.response.text[:200]}"
        return {"error": error_details}
    except Exception as e:
        app_logger.error(f"Azure summarization error: {e}\n{traceback.format_exc()}")
        return {"error": f"An unexpected error occurred during summarization: {str(e)}"}

def transcribe_audio_azure(audio_bytesio_object, input_filename="temp_audio.wav"):
    if not speech_config:
        app_logger.error("transcribe_audio_azure called but speech_config is not initialized.")
        return {"error": "Speech service not available or not configured."}, ""
    
    app_logger.info(f"Speech Service Key being used (last 4 chars): ...{SPEECH_KEY[-4:] if SPEECH_KEY and len(SPEECH_KEY) >=4 else 'N/A'}")
    app_logger.info(f"Speech Service Region being used: {SPEECH_REGION}")

    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        try:
            os.makedirs(upload_folder)
            app_logger.info(f"Created upload folder for temp audio: {upload_folder}")
        except OSError as e:
            app_logger.error(f"Could not create upload folder {upload_folder}: {e}")
            return {"error": f"Could not create temporary directory for audio processing: {e}"}, ""
            
    temp_audio_path = os.path.join(upload_folder, secure_filename(input_filename))
    
    speech_recognizer = None # Initialize to None
    audio_input = None # Initialize to None

    try:
        with open(temp_audio_path, 'wb') as f_temp:
            f_temp.write(audio_bytesio_object.getvalue())
        
        app_logger.info(f"Temporary audio file saved at: {temp_audio_path}")
        audio_input = speechsdk.AudioConfig(filename=temp_audio_path)
        
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
        app_logger.info(f"Starting Azure speech recognition for {temp_audio_path}...")
        
        result = speech_recognizer.recognize_once_async().get() 
        full_transcript = ""

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            full_transcript = result.text
            app_logger.info(f"Recognition successful. Transcript snippet: '{full_transcript[:100]}...'")
        elif result.reason == speechsdk.ResultReason.NoMatch:
            app_logger.warning(f"No speech could be recognized from {temp_audio_path}. Details: {result.no_match_details}")
            return {"error": "No speech could be recognized from the audio."}, ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            app_logger.error(f"Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                app_logger.error(f"Cancellation ErrorCode: {cancellation_details.error_code}")
                app_logger.error(f"Cancellation ErrorDetails: {cancellation_details.error_details}")
            if "AuthenticationFailure" in cancellation_details.error_details:
                 return {"error": "Speech service authentication failed. Check key/region."}, ""
            return {"error": f"Speech recognition canceled: {cancellation_details.reason}. Details: {cancellation_details.error_details}"}, ""
        
        # Explicitly delete/close Azure SDK objects
        if speech_recognizer:
            del speech_recognizer 
            speech_recognizer = None 
        if audio_input:
            del audio_input
            audio_input = None

        return {"transcript": full_transcript}, full_transcript
    except Exception as e: 
        app_logger.error(f"Error during Azure STT for file {temp_audio_path}: {e}\n{traceback.format_exc()}")
        if "SPXERR_AUDIO_SYS_ERROR" in str(e) or "SPXERR_FILE_OPEN_FAILED" in str(e):
            return {"error": f"Audio system or file error during transcription: {str(e)}"}, ""
        elif "SPXERR_CONNECTION_FAILURE" in str(e):
            return {"error": f"Network connection failure during transcription: {str(e)}"}, ""
        return {"error": f"An error occurred during transcription: {str(e)}"}, ""
    finally:
        # Ensure deletion in finally block as well, in case of exceptions before explicit del
        if speech_recognizer is not None: 
            del speech_recognizer
        if audio_input is not None:
            del audio_input
            
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                app_logger.info(f"Temporary audio file {temp_audio_path} deleted.")
            except Exception as e_del:
                app_logger.error(f"Could not delete temp audio file {temp_audio_path}: {e_del}")

def analyze_image_azure(image_stream_bytesio, features_list=["Description", "Tags"]):
    if not computervision_client:
        app_logger.error("analyze_image_azure called but computervision_client is not initialized.")
        return {"error": "Vision service not available or not configured."}
    try:
        visual_feature_types_enum = []
        feature_map = { 
            "Description": VisualFeatureTypes.description, "Tags": VisualFeatureTypes.tags,
            "Objects": VisualFeatureTypes.objects, "Faces": VisualFeatureTypes.faces,
            "ImageType": VisualFeatureTypes.image_type, "Color": VisualFeatureTypes.color,
            "Adult": VisualFeatureTypes.adult, "Brands": VisualFeatureTypes.brands,
            "Categories": VisualFeatureTypes.categories
        }
        for feature_name in features_list:
            if feature_name in feature_map:
                visual_feature_types_enum.append(feature_map[feature_name])
        
        if not visual_feature_types_enum: 
            visual_feature_types_enum = [VisualFeatureTypes.description]

        image_stream_bytesio.seek(0) 
        analysis = computervision_client.analyze_image_in_stream(image_stream_bytesio, visual_features=visual_feature_types_enum)
        
        result = {"analysis_details": {}} 

        if VisualFeatureTypes.description in visual_feature_types_enum and analysis.description and analysis.description.captions:
            result["description"] = [caption.text for caption in analysis.description.captions]
            if analysis.description.tags: 
                 result["analysis_details"]["description_tags"] = analysis.description.tags
        
        if VisualFeatureTypes.tags in visual_feature_types_enum and analysis.tags:
            result["tags"] = [{"name": tag.name, "confidence": f"{tag.confidence:.2f}"} for tag in analysis.tags]
        
        if VisualFeatureTypes.objects in visual_feature_types_enum and analysis.objects:
            result["objects"] = [{"object": obj.object_property, 
                                  "confidence": f"{obj.confidence:.2f}",
                                  "rectangle": obj.rectangle.__dict__ if obj.rectangle else None} 
                                 for obj in analysis.objects]
        
        if VisualFeatureTypes.faces in visual_feature_types_enum and analysis.faces:
            result["faces"] = [{"age": face.age, 
                                "gender": str(face.gender), 
                                "rectangle": face.face_rectangle.__dict__} 
                               for face in analysis.faces]
        
        if VisualFeatureTypes.categories in visual_feature_types_enum and analysis.categories:
            result["categories"] = [{"name": cat.name, "score": f"{cat.score:.2f}"} for cat in analysis.categories]
        
        if VisualFeatureTypes.adult in visual_feature_types_enum and analysis.adult:
            result["adult_content"] = {
                "is_adult_content": analysis.adult.is_adult_content, 
                "adult_score": f"{analysis.adult.adult_score:.2f}",
                "is_racy_content": analysis.adult.is_racy_content, 
                "racy_score": f"{analysis.adult.racy_score:.2f}",
                "is_gory_content": analysis.adult.is_gory_content if hasattr(analysis.adult, 'is_gory_content') else None, 
                "gore_score": f"{analysis.adult.gore_score:.2f}" if hasattr(analysis.adult, 'gore_score') else None 
            }

        if not any(val for key, val in result.items() if key != "analysis_details" or (isinstance(val, dict) and any(val.values()))):
             app_logger.warning(f"Image analysis returned no specific features for the request. Raw output keys: {analysis.__dict__.keys()}")
             return {"message": "No specific features detected by the analysis or an issue with the response.", "raw_analysis_output_keys": list(analysis.__dict__.keys())}
        
        return result

    except ClientAuthenticationError as cae:
        app_logger.error(f"Azure Vision authentication error: {cae}\n{traceback.format_exc()}")
        return {"error": "Vision service authentication failed. Check your subscription key and endpoint."}
    except ServiceRequestError as sre:
        app_logger.error(f"Azure Vision service request error: {sre}\n{traceback.format_exc()}")
        return {"error": f"Could not connect to Vision service: {sre.message}. Check endpoint and network."}
    except HttpResponseError as hre:
        app_logger.error(f"Azure Vision HTTP error: {hre}\n{traceback.format_exc()}")
        error_message = f"Vision service request failed: {hre.message} (Status: {hre.status_code})."
        if hre.response and hasattr(hre.response, 'text') and hre.response.text:
            try:
                error_content = json.loads(hre.response.text)
                if 'error' in error_content and 'message' in error_content['error']:
                    error_message += f" Azure details: {error_content['error']['message']}"
                    if 'innererror' in error_content['error'] and 'message' in error_content['error']['innererror']:
                         error_message += f" Inner details: {error_content['error']['innererror']['message']}"
            except json.JSONDecodeError:
                pass 
        return {"error": error_message, "status_code": hre.status_code}
    except Exception as e:
        app_logger.error(f"Azure image analysis error: {e}\n{traceback.format_exc()}")
        if "Invalid URL" in str(e) or "'Endpoint'" in str(e): 
             return {"error": "Vision service client failed to initialize, likely due to an invalid 'VISION_ENDPOINT' in the .env file or configuration. Please verify the endpoint URL."}
        return {"error": f"An unexpected error occurred during image analysis: {str(e)}"}


# --- PDF Export ---
def export_to_pdf_reportlab(text_content, title="Summary"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles['h1']), Spacer(1, 0.2*inch)]

    paragraphs = str(text_content).split('\n')
    for para_text in paragraphs:
        if para_text.strip():
            story.append(Paragraph(para_text, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Spacer(1, 0.05*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('index_page'))

@app.route('/toolkit')
def index_page():
    return render_template('index.html', current_user=current_user)

@app.route('/image-tools')
def image_analyzer_page_route():
    return render_template('image_analyzer.html', current_user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'warning')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Error during registration: {e}")
            flash('Registration failed due to a server error. Please try again.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index_page'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=request.form.get('remember_me') == 'on')
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index_page'))
        else:
            flash('Login failed. Check username and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/my-summaries')
@login_required
def my_summaries():
    user_summaries = Summary.query.filter_by(user_id=current_user.id).order_by(Summary.timestamp.desc()).all()
    return render_template('my_summaries.html', summaries=user_summaries, current_user=current_user)

@app.route('/delete_summary/<int:summary_id>', methods=['POST'])
@login_required
def delete_summary(summary_id):
    summary_to_delete = Summary.query.get_or_404(summary_id)
    if summary_to_delete.author != current_user:
        flash("You are not authorized to delete this summary.", "danger")
        return redirect(url_for('my_summaries'))
    try:
        db.session.delete(summary_to_delete)
        db.session.commit()
        flash('Summary deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        app_logger.error(f"Error deleting summary {summary_id}: {e}")
        flash('Failed to delete summary due to a server error.', 'danger')
    return redirect(url_for('my_summaries'))

@app.route('/export_summary_pdf/<int:summary_id>')
@login_required
def export_summary_pdf_route(summary_id):
    summary_obj = Summary.query.get_or_404(summary_id)
    if summary_obj.author != current_user:
        flash("Not authorized to export this summary.", "danger")
        return redirect(url_for('my_summaries'))

    pdf_title = f"Summary: {summary_obj.input_type[:30]}"
    pdf_buffer = export_to_pdf_reportlab(summary_obj.summarized_text, title=pdf_title)

    return Response(pdf_buffer.getvalue(),
                    mimetype='application/pdf',
                    headers={"Content-Disposition": f"attachment;filename=summary_{summary_obj.id}.pdf"})

# --- API/Processing Routes ---
@app.route('/process-text-content', methods=['POST'])
def process_text_content():
    data = request.form
    input_type = data.get('inputType')
    action = data.get('actionType')
    text_content = ""
    original_filename_or_title = input_type # Default
    db_input_type_name = input_type # For saving to DB, more specific

    app_logger.info(f"Processing request: inputType='{input_type}', actionType='{action}'")

    # --- Input Handling ---
    if input_type == 'audio_file':
        if action != 'transcribe_audio':
            app_logger.warning(f"Input type is 'audio_file' but action is '{action}'. Forcing action to 'transcribe_audio'.")
            action = 'transcribe_audio'

    if input_type == 'text':
        text_content = data.get('textInput')
        original_filename_or_title = "Direct Text Input"
        db_input_type_name = "Direct Text Input"
    elif input_type == 'pdf':
        if 'pdfFile' not in request.files: return jsonify({"error": "No PDF file part."}), 400
        file = request.files['pdfFile']
        if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS_PDF):
            return jsonify({"error": "Invalid PDF file or no file selected."}), 400
        original_filename_or_title = secure_filename(file.filename)
        db_input_type_name = f"PDF: {original_filename_or_title}"
        text_content = extract_pdf_text_pypdf2(file.stream)
        if text_content is None: return jsonify({"error": "Could not extract text from PDF."}), 500

    elif input_type == 'youtube':
        youtube_url = data.get('youtubeUrl')
        if not youtube_url: return jsonify({"error": "YouTube URL missing."}), 400

        # Attempt 1: Use youtube_transcript_api
        transcript_text_api, video_title_api, error_msg_api = get_transcript_from_youtube_api(youtube_url)

        current_video_title = video_title_api if video_title_api else "YouTube Video"


        if transcript_text_api:
            text_content = transcript_text_api
            original_filename_or_title = current_video_title
            db_input_type_name = f"YouTube: {current_video_title}"
            app_logger.info(f"Successfully obtained transcript via youtube_transcript_api for {youtube_url}")
        else:
            app_logger.warning(f"youtube_transcript_api failed for {youtube_url}: {error_msg_api}. Falling back to Pytube audio download and Azure STT.")
            # Fallback to Pytube
            audio_buffer, video_title_pytube, error_msg_pytube = get_youtube_audio_stream_pytube(youtube_url)

            if video_title_pytube and (not current_video_title or current_video_title == "YouTube Video (Transcript API)" or "Error" in current_video_title):
                current_video_title = video_title_pytube

            if error_msg_pytube:
                return jsonify({"error": f"YouTube processing error (Pytube fallback for '{current_video_title}'): {error_msg_pytube}"}), 500
            if not audio_buffer:
                return jsonify({"error": f"Failed to get audio stream from YouTube for '{current_video_title}' (Pytube fallback)."}), 500

            original_filename_or_title = current_video_title
            db_input_type_name = f"YouTube (Fallback STT): {current_video_title}"

            stt_result, transcript_text_stt = transcribe_audio_azure(audio_buffer, input_filename=f"youtube_{secure_filename(current_video_title[:20])}.wav")
            if "error" in stt_result:
                return jsonify({"error": f"Azure STT error for YouTube audio from '{current_video_title}' (Pytube fallback): {stt_result['error']}"}), 500
            text_content = transcript_text_stt
            app_logger.info(f"Successfully obtained transcript via Pytube + Azure STT for {youtube_url} (Title: {current_video_title})")

        if action == 'transcribe_audio':
             return jsonify({"transcript": text_content, "message": f"YouTube video '{current_video_title}' processed successfully."})


    elif input_type == 'audio_file':
        # For direct audio file uploads, text_content will be populated during the 'transcribe_audio' action.
        pass
    else:
        app_logger.error(f"Invalid inputType received in main processing block: {input_type}")
        return jsonify({"error": "Invalid input type specified in request."}), 400

    # --- Action Handling ---
    if not text_content and not (action == 'transcribe_audio' and input_type == 'audio_file'):
         app_logger.warning(f"No text content available for action '{action}' with inputType '{input_type}'.")
         if input_type != 'youtube':
             return jsonify({"error": "No text content to process for the selected action."}), 400

    result = {}
    target_language = data.get('targetLanguage', 'en')
    source_language = data.get('sourceLanguage')

    if action == 'translate':
        if not text_content: return jsonify({"error": "No text provided for translation."}), 400
        result = translate_text_azure(text_content, target_languages=[target_language], source_language=source_language if source_language else None)
    elif action == 'sentiment':
        if not text_content: return jsonify({"error": "No text provided for sentiment analysis."}), 400
        result = analyze_text_sentiment_azure(text_content)
    elif action == 'summarize_abstractive':
        if not text_content: return jsonify({"error": "No text provided for abstractive summarization."}), 400
        result = summarize_text_azure(text_content, kind="abstractive", max_sentence_count=int(data.get('sentenceCount', 3)))
    elif action == 'summarize_extractive':
        if not text_content: return jsonify({"error": "No text provided for extractive summarization."}), 400
        result = summarize_text_azure(text_content, kind="extractive", max_sentence_count=int(data.get('sentenceCount', 3)))
    elif action == 'transcribe_audio':
        if input_type == 'audio_file':
            if 'audioFile' not in request.files:
                app_logger.error("Action is 'transcribe_audio' with inputType 'audio_file' but no 'audioFile' in request.files.")
                return jsonify({"error": "No audio file part for transcription."}), 400
            audio_file = request.files['audioFile']
            if audio_file.filename == '' or not allowed_file(audio_file.filename, ALLOWED_EXTENSIONS_AUDIO):
                return jsonify({"error": "Invalid audio file or no file selected for transcription."}), 400

            original_filename_or_title = secure_filename(audio_file.filename)
            db_input_type_name = f"Audio File: {original_filename_or_title}"
            audio_buffer = BytesIO(audio_file.read())
            stt_result, transcript_text = transcribe_audio_azure(audio_buffer, input_filename=original_filename_or_title)

            if "error" in stt_result: return jsonify({"error": f"Azure STT error for audio file: {stt_result['error']}"}), 500
            result = {"transcript": transcript_text}
            text_content = transcript_text # Update text_content with the transcript for saving
        elif input_type == 'youtube':
             result = {"transcript": text_content, "message": f"YouTube video '{original_filename_or_title}' transcript was already processed."}
        else:
            return jsonify({"error": "Transcription called with an unexpected configuration."}), 400
    else:
        app_logger.error(f"Invalid actionType received: {action}")
        return jsonify({"error": "Invalid action type specified."}), 400

    # --- Save to Database (if applicable and user logged in) ---
    if current_user.is_authenticated:
        processed_text_to_save = None
        if 'summarize' in action and 'summary' in result and not result.get('error'):
            processed_text_to_save = result.get('summary')
        elif action == 'transcribe_audio' and 'transcript' in result and not result.get('error'):
            processed_text_to_save = result.get('transcript')

        if processed_text_to_save:
            try:
                original_text_for_db = text_content
                if action == 'transcribe_audio':
                    original_text_for_db = f"Source: {db_input_type_name}"
                elif input_type == 'youtube' and 'summarize' in action:
                     original_text_for_db = f"Transcript from: {db_input_type_name}"


                new_db_entry = Summary(
                    original_text=original_text_for_db[:20000],
                    summarized_text=processed_text_to_save,
                    input_type=db_input_type_name,
                    user_id=current_user.id,
                    language=target_language if action == 'translate' else (source_language if source_language and action == 'translate' else 'en')
                )
                db.session.add(new_db_entry)
                db.session.commit()
                result['db_summary_id'] = new_db_entry.id
                result['save_status'] = f"{'Summary' if 'summarize' in action else 'Transcript'} saved for '{db_input_type_name}'."
            except Exception as e:
                db.session.rollback()
                app_logger.error(f"Error saving result to DB for {db_input_type_name}: {e}\n{traceback.format_exc()}")
                result['save_status_error'] = "Could not save result to database due to a server error."

    return jsonify(result)


@app.route('/image-analyze-azure', methods=['POST'])
def analyze_image_route_azure():
    if not computervision_client:
        return jsonify({"error": "Image analysis service is not available or not configured."}), 503

    if 'imageFile' not in request.files: return jsonify({"error": "No image file part."}), 400
    file = request.files['imageFile']
    analysis_type_from_ui = request.form.get('analysisType', 'captions')

    if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS_IMG):
        return jsonify({"error": "Invalid image file or no file selected."}), 400

    image_stream_bytesio = BytesIO(file.read())
    features_to_request = []

    if analysis_type_from_ui == 'captions': features_to_request = ["Description"]
    elif analysis_type_from_ui == 'tags': features_to_request = ["Tags"]
    elif analysis_type_from_ui == 'objects': features_to_request = ["Objects"]
    elif analysis_type_from_ui == 'fullAnalysis':
        features_to_request = ["Description", "Tags", "Objects", "Faces", "ImageType", "Color", "Adult", "Brands", "Categories"]
    else:
        features_to_request = ["Description"]
        app_logger.warning(f"Unknown image analysisType '{analysis_type_from_ui}', defaulting to Description.")

    result = analyze_image_azure(image_stream_bytesio, features_list=features_to_request)
    return jsonify(result)

# --- Main Execution & Setup ---
def create_app_directories():
    """Creates necessary directories if they don't exist."""
    # For Docker, these paths will be relative to /app
    dirs_to_create = [
        app.config['UPLOAD_FOLDER'],
        os.path.join(app.root_path, 'instance'),
        nltk_data_dir # Use the globally defined nltk_data_dir
    ]
    for d in dirs_to_create:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                app_logger.info(f"Created directory: {d}")
            except OSError as e:
                 app_logger.error(f"Could not create directory {d}: {e}")


def initialize_database(flask_app):
    """Creates database tables if they don't exist."""
    with flask_app.app_context():
        try:
            db.create_all()
            app_logger.info("Database tables checked/created successfully.")
        except Exception as e:
            app_logger.error(f"Error during db.create_all(): {e}\n{traceback.format_exc()}")
            app_logger.error("Please ensure your DATABASE_URL is correctly configured and the database server is accessible.")


if __name__ == '__main__':
    # This block is primarily for local development when running `python app.py`
    # It will not be executed when Gunicorn runs the app in Docker.
    logging.getLogger("werkzeug").setLevel(logging.WARNING) 
    logging.getLogger("pytube").setLevel(logging.WARNING) 
    
    create_app_directories() # Ensure directories are created for local dev
    initialize_database(app) 
    
    app_logger.info("Azure AI Mega Toolkit starting on Flask development server (local)...")
    # Use a different port for local "python app.py" to avoid conflict if also running Docker locally on 8000
    local_run_port = 5000
    is_development = os.environ.get('FLASK_ENV', 'production').lower() == 'development'
    app_logger.info(f"Flask FLASK_ENV is '{os.environ.get('FLASK_ENV', 'production')}', running with debug={is_development} on port {local_run_port}")
    
    app.run(debug=is_development, host='0.0.0.0', port=local_run_port)
