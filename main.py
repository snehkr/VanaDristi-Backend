# VanaDristi API - Version 2.2

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import json
import time
import asyncio
import hashlib
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List

# --- FastAPI & Related ---
from fastapi.params import Query
import uvicorn
import httpx
from fastapi import (
    FastAPI,
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Asynchronous Database (Motor) ---
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from pymongo import DESCENDING

# --- Pydantic Schemas ---
from pydantic import BaseModel, Field

# --- Rate Limiting ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Google Gemini AI ---
import google.generativeai as genai

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
load_dotenv()

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "vana_dristi_ai")

# --- Gemini AI Configuration ---
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- ImageKit.io Configuration ---
IMAGEKIT_URL = os.getenv("IMAGEKIT_URL", "https://ik.imagekit.io/vanadristi")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_UPLOAD_FOLDER = os.getenv("IMAGEKIT_UPLOAD_FOLDER", "VanaDristi")

# --- JWT Authentication Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# --- Telegram Notification Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- File Upload Validation ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ==============================================================================
# 3. INITIALIZATION (DB, Security, Rate Limiter)
# ==============================================================================

# --- Database Client ---
db_client = AsyncIOMotorClient(MONGO_URI)
db = db_client[DB_NAME]

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)


# ==============================================================================
# 4. PYDANTIC MODELS (SCHEMAS)
# ==============================================================================


# --- Plant Models ---
class AlertThresholds(BaseModel):
    min_soil_moisture: Optional[int] = Field(None, ge=0, le=100)
    max_soil_moisture: Optional[int] = Field(None, ge=0, le=100)
    min_temperature: Optional[int] = Field(None)
    max_temperature: Optional[int] = Field(None)
    min_humidity: Optional[int] = Field(None, ge=0, le=100)
    max_humidity: Optional[int] = Field(None, ge=0, le=100)
    min_light_intensity: Optional[int] = Field(None)
    max_light_intensity: Optional[int] = Field(None)
    min_leaf_color_index: Optional[int] = Field(None)
    max_leaf_color_index: Optional[int] = Field(None)


class PlantBase(BaseModel):
    name: str = Field(..., max_length=100)
    species: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=100)
    telegram_chat_id: Optional[str] = Field(
        None, description="Telegram Chat ID for receiving alerts."
    )
    alerts: AlertThresholds = Field(default_factory=AlertThresholds)


class PlantCreate(PlantBase):
    pass


class PlantUpdate(PlantBase):
    pass


class PlantInDB(PlantBase):
    id: str = Field(alias="_id")
    created_at: datetime


# --- Sensor & AI Models ---
class SensorIn(BaseModel):
    plant_id: str
    soil_moisture: Optional[float]
    temperature: Optional[float]
    humidity: Optional[float]
    light: Optional[float]
    leaf_color: Optional[float]
    image_url: Optional[str] = None


class ChatQuery(BaseModel):
    question: str
    plant_id: str


# ==============================================================================
# 5. CORE SERVICE HELPERS (Gemini, ImageKit, Email)
# ==============================================================================

# --- Gemini Service ---
_gemini_model = None


def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _gemini_model


async def analyze_text(prompt: str, response_format: str = "json") -> str:
    try:
        model = get_gemini_model()
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "response_mime_type": (
                    "application/json" if response_format == "json" else "text/plain"
                ),
            },
        )
        return response.text
    except Exception as e:
        raise


async def analyze_with_image(prompt: str, image_bytes: bytes) -> str:
    try:
        model = get_gemini_model()
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}
        response = await model.generate_content_async(
            [prompt, image_part],
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
            },
        )
        return response.text
    except Exception as e:
        raise


# --- ImageKit Service ---
async def upload_to_imagekit(
    file_content: bytes, original_filename: str
) -> Dict[str, Any]:
    final_filename = f"{hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]}.{original_filename.split('.')[-1]}"
    files = {"file": (final_filename, file_content)}
    data = {
        "fileName": final_filename,
        "useUniqueFileName": "false",
        "folder": IMAGEKIT_UPLOAD_FOLDER,
    }
    auth = httpx.BasicAuth(IMAGEKIT_PRIVATE_KEY, "")
    final_resp = {}
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://upload.imagekit.io/api/v1/files/upload",
            auth=auth,
            files=files,
            data=data,
        )
        response.raise_for_status()
        if response.status_code == 200:
            final_resp["status"] = "ok"
            final_resp["message"] = "Image uploaded successfully."
            final_resp["image_url"] = (
                IMAGEKIT_URL + "/" + IMAGEKIT_UPLOAD_FOLDER + "/" + final_filename
            )
        else:
            final_resp["status"] = "error"
            final_resp["message"] = "Failed to upload image."

    return final_resp


# --- Telegram Notification Service ---
async def send_telegram_alert(chat_id: str, message: str):
    if not TELEGRAM_BOT_TOKEN:
        print("WARN: TELEGRAM_BOT_TOKEN not set. Skipping Telegram alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            print(f"Alert sent to Telegram chat ID {chat_id}")
        except httpx.HTTPStatusError as e:
            print(f"ERROR: Failed to send Telegram alert: {e.response.text}")


# --- Background Task Logic ---
async def check_alerts_and_notify(plant_id: str, sensor_data: dict):
    plant = await db.plants.find_one({"_id": ObjectId(plant_id)})

    chat_id = TELEGRAM_CHAT_ID
    if not plant or not chat_id:
        return

    alerts = plant.get("alerts", {})
    messages = []

    # Check all thresholds
    if (
        (t := alerts.get("min_soil_moisture")) is not None
        and (s := sensor_data.get("soil_moisture")) is not None
        and s < t
    ):
        messages.append(
            f"💧 Soil moisture is critically low ({s}%), below your threshold of {t}%."
        )
    if (
        (t := alerts.get("max_soil_moisture")) is not None
        and (s := sensor_data.get("soil_moisture")) is not None
        and s > t
    ):
        messages.append(
            f"💧 Soil moisture is too high ({s}%), above your threshold of {t}%."
        )
    if (
        (t := alerts.get("min_temperature")) is not None
        and (s := sensor_data.get("temperature")) is not None
        and s < t
    ):
        messages.append(
            f"🌡️ Temperature is too low ({s}°C), below your threshold of {t}°C."
        )
    if (
        (t := alerts.get("max_temperature")) is not None
        and (s := sensor_data.get("temperature")) is not None
        and s > t
    ):
        messages.append(
            f"🌡️ Temperature is too high ({s}°C), above your threshold of {t}°C."
        )
    if (
        (t := alerts.get("min_humidity")) is not None
        and (s := sensor_data.get("humidity")) is not None
        and s < t
    ):
        messages.append(
            f"💨 Humidity is critically low ({s}%), below your threshold of {t}%."
        )
    if (
        (t := alerts.get("max_humidity")) is not None
        and (s := sensor_data.get("humidity")) is not None
        and s > t
    ):
        messages.append(
            f"💨 Humidity is too high ({s}%), above your threshold of {t}%."
        )
    if (
        (t := alerts.get("min_light_intensity")) is not None
        and (s := sensor_data.get("light")) is not None
        and s < t
    ):
        messages.append(
            f"☀️ Light intensity is too low ({s} lux), below your threshold of {t} lux."
        )
    if (
        (t := alerts.get("max_light_intensity")) is not None
        and (s := sensor_data.get("light")) is not None
        and s > t
    ):
        messages.append(
            f"☀️ Light intensity is too high ({s} lux), above your threshold of {t} lux."
        )

    if messages:
        full_message = f"🚨 *Plant Alert for '{plant['name']}'* 🚨\n\n" + "\n".join(
            messages
        )
        await send_telegram_alert(chat_id, full_message)


# ==============================================================================
# 6. FASTAPI LIFESPAN & APP INITIALIZATION
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    # Add indexes on startup
    await db.plants.create_index("name")
    await db.sensor_data.create_index([("plant_id", 1), ("timestamp", -1)])
    await db.chat_history.create_index([("plant_id", 1), ("timestamp", -1)])
    print("Database indexes ensured.")
    yield
    print("Application shutdown.")
    db_client.close()


app = FastAPI(
    title="VanaDristi API",
    description="A public platform for monitoring plant health with AI and Telegram alerts.",
    version="2.2.0",
    lifespan=lifespan,
)

# --- Middleware ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 7. API ROUTERS
# ==============================================================================

# ------------------------------------------------------------------------------
# 7.1 Home Router
# ------------------------------------------------------------------------------
home_router = APIRouter(tags=["Home"])


@home_router.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Welcome to the VanaDristi API Platform!",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ------------------------------------------------------------------------------
# 7.2 Plant Management Router
# ------------------------------------------------------------------------------
plant_router = APIRouter(prefix="/api/v1/plants", tags=["Plant Management"])


@plant_router.post("/", response_model=PlantInDB)
async def create_plant(plant: PlantCreate):
    plant_doc = plant.model_dump()
    plant_doc["created_at"] = datetime.utcnow()
    result = await db.plants.insert_one(plant_doc)
    created_plant = await db.plants.find_one({"_id": result.inserted_id})
    created_plant["_id"] = str(created_plant["_id"])
    return PlantInDB(**created_plant)


@plant_router.get("/", response_model=List[PlantInDB])
async def get_all_plants(limit: int = 100):
    plants_cursor = db.plants.find().limit(limit)
    plants = await plants_cursor.to_list(length=limit)
    for p in plants:
        p["_id"] = str(p["_id"])
    return [PlantInDB(**p) for p in plants]


@plant_router.get("/{plant_id}", response_model=PlantInDB)
async def get_plant(plant_id: str):
    plant = await db.plants.find_one({"_id": ObjectId(plant_id)})
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    plant["_id"] = str(plant["_id"])
    return PlantInDB(**plant)


@plant_router.put("/{plant_id}", response_model=PlantInDB)
async def update_plant(plant_id: str, plant_update: PlantUpdate):
    await db.plants.update_one(
        {"_id": ObjectId(plant_id)},
        {"$set": plant_update.model_dump(exclude_unset=True)},
    )
    updated_plant = await db.plants.find_one({"_id": ObjectId(plant_id)})
    if not updated_plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    updated_plant["_id"] = str(updated_plant["_id"])
    return PlantInDB(**updated_plant)


@plant_router.delete("/{plant_id}", status_code=204)
async def delete_plant(plant_id: str):
    result = await db.plants.delete_one({"_id": ObjectId(plant_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Plant not found")


# ------------------------------------------------------------------------------
# 7.2 Sensor Data Router
# ------------------------------------------------------------------------------
sensor_router = APIRouter(prefix="/api/v1/sensor", tags=["Sensor Data"])


@sensor_router.post("/upload")
@limiter.limit("40/minute")
async def upload_sensor_data(
    request: Request,
    background_tasks: BackgroundTasks,
    plant_id: str = Form(...),
    sensor_json: str = Form(...),
    image: Optional[UploadFile] = File(None),
):
    plant = await db.plants.find_one({"_id": ObjectId(plant_id)})
    if not plant:
        raise HTTPException(
            status_code=404, detail="Plant not found. Please create the plant first."
        )

    data = json.loads(sensor_json)
    data["plant_id"] = plant_id
    data["plant_name"] = plant.get("name", "Unknown Plant")
    data["plant_species"] = plant.get("species", "Unknown Species")
    data["timestamp"] = datetime.utcnow()

    if image:
        contents = await image.read()
        imagekit_res = await upload_to_imagekit(contents, image.filename)
        if imagekit_res.get("status") == "ok":
            data["image_url"] = imagekit_res.get("image_url")

    await db.sensor_data.insert_one(data)
    background_tasks.add_task(check_alerts_and_notify, plant_id, data)

    return JSONResponse(status_code=202, content={"message": "Sensor data accepted."})


@sensor_router.post("/upload_first")
@limiter.limit("40/minute")
async def upload_first_sensor_data(
    request: Request,
    background_tasks: BackgroundTasks,
    sensor_json: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Uploads sensor data and an image.
    The image is used to identify the plant via AI. If the plant doesn't exist,
    it's created with default settings. The sensor data is then associated
    with this plant.
    """
    # Analyze the image to identify the plant
    contents = await image.read()
    prompt = f"""
    You are an expert botanist and agricultural AI. Your task is to identify a plant from an image and generate a complete, realistic default profile for it in JSON format.

    **Context:**
    - Current Location: Jaipur, Rajasthan, India
    - Current Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    **Instructions:**
    1.  Identify the plant in the image.
    2.  Based on the identified species, generate a JSON object containing a plausible default care profile.
    3.  The `alerts` values should be sensible, general-purpose thresholds appropriate for this plant species. For example, a cactus will have different water needs than a fern.
    4.  Do not include any text, notes, or markdown formatting before or after the JSON object.

    **Required JSON Output Structure:**
    {{
      "common_name": "string",
      "scientific_name": "string",
      "location": "string",
      "alerts": {{
        "min_soil_moisture": integer,
        "max_soil_moisture": integer,
        "min_temperature": integer,
        "max_temperature": integer,
        "min_humidity": integer,
        "max_humidity": integer,
        "min_light_intensity": integer,
        "max_light_intensity": integer,
        "min_leaf_color_index": null,
        "max_leaf_color_index": null
      }}
    }}
    """
    try:
        ai_result_text = await asyncio.wait_for(
            analyze_with_image(prompt, contents), timeout=20.0
        )
        parsed_result = json.loads(ai_result_text)
        plant_name = parsed_result.get("common_name")
        if not plant_name:
            raise ValueError("AI response did not contain a 'common_name'")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI analysis timed out.")
    except (Exception, ValueError) as e:
        raise HTTPException(
            status_code=502,
            detail=f"AI model call for identification failed or returned invalid data: {e}",
        )

    # Check for an existing plant or create a new one
    existing_plant = await db.plants.find_one({"name": plant_name})
    plant_created = False

    if existing_plant:
        plant_id = str(existing_plant["_id"])
        plant = existing_plant
    else:
        # Use the detailed profile generated by the AI
        new_plant_data = {
            "name": parsed_result.get("common_name"),
            "species": parsed_result.get("scientific_name"),
            "location": parsed_result.get("location", "Unknown Location"),
            "telegram_chat_id": "",
            "alerts": parsed_result.get("alerts", {}),
            "created_at": datetime.utcnow(),
        }
        try:
            result = await db.plants.insert_one(new_plant_data)
            plant_id = str(result.inserted_id)
            plant = await db.plants.find_one({"_id": ObjectId(plant_id)})
            plant_created = True
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to create new plant in database: {e}"
            )

    if not plant:
        raise HTTPException(
            status_code=500,
            detail="Could not create or find a plant for the uploaded data.",
        )

    # Process and save sensor data
    data = json.loads(sensor_json)
    data["plant_id"] = plant_id
    data["plant_name"] = plant.get("name")
    data["plant_species"] = plant.get("species")
    data["timestamp"] = datetime.utcnow()

    # Upload image to ImageKit
    try:
        imagekit_res = await upload_to_imagekit(contents, image.filename)
        if imagekit_res.get("status") == "ok":
            data["image_url"] = imagekit_res.get("image_url")
    except Exception as e:
        print(f"WARN: Failed to upload image to ImageKit: {e}")
        data["image_url"] = None

    # Save sensor data and trigger background tasks
    await db.sensor_data.insert_one(data)
    background_tasks.add_task(check_alerts_and_notify, plant_id, data)

    return JSONResponse(
        status_code=202,
        content={
            "message": "Sensor data accepted.",
            "plant_id": plant_id,
            "plant_name": plant.get("name"),
            "plant_created": plant_created,
        },
    )


@sensor_router.get("/history/{plant_id}")
async def get_sensor_history(plant_id: str, limit: int = 100):
    cursor = (
        db.sensor_data.find({"plant_id": plant_id}, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    results = await cursor.to_list(length=limit)
    return json.loads(json.dumps(results, default=str))


@sensor_router.get("/latest/{plant_id}")
async def get_latest_sensor_data(plant_id: str):
    latest_item = await db.sensor_data.find_one(
        {"plant_id": plant_id}, {"_id": 0}, sort=[("timestamp", DESCENDING)]
    )
    if not latest_item:
        raise HTTPException(
            status_code=404, detail="No sensor data found for this plant."
        )
    return json.loads(json.dumps(latest_item, default=str))


@sensor_router.get("/trends/{plant_id}")
@limiter.limit("45/minute")
async def get_plant_trends(request: Request, plant_id: str):
    pipeline = [
        {"$match": {"plant_id": plant_id}},
        {
            "$project": {
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "temperature": "$temperature",
                "soil_moisture": "$soil_moisture",
                "humidity": "$humidity",
            }
        },
        {
            "$group": {
                "_id": "$day",
                "avg_temp": {"$avg": "$temperature"},
                "avg_moisture": {"$avg": "$soil_moisture"},
                "avg_humidity": {"$avg": "$humidity"},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    trends = await db.sensor_data.aggregate(pipeline).to_list(length=365)
    return trends


# ------------------------------------------------------------------------------
# 7.3 AI Analysis Router
# ------------------------------------------------------------------------------
ai_router = APIRouter(prefix="/api/v1/ai", tags=["AI Analysis"])


@ai_router.post("/identify")
@limiter.limit("45/minute")
async def identify_plant_species(
    request: Request,
    image: UploadFile = File(...),
):
    contents = await image.read()
    prompt = """
You are a professional botanist. Analyze the given plant image and respond strictly in JSON with these keys:
- 'common_name': widely used English or regional name.
- 'scientific_name': correct botanical name (Genus + species).
- 'family': botanical family name.
- 'origin': native region or country.
- 'growth_habit': tree, shrub, herb, climber, etc.
- 'flowering_season': typical flowering months.
- 'edible_or_medicinal': short note if it has edible/medicinal use.
- 'uses': short list of ornamental, medicinal, timber, or other uses.
- 'care_summary': concise guide (sunlight, watering, soil, special tips).
- 'common_diseases': list of major diseases/pests (if any) with a one-line prevention/treatment tip.
- 'diagnosis_from_image': if this plant shows visible disease/pest symptoms in the uploaded photo, name the issue and give a short remedy. If healthy, return "No visible disease detected."
Do not include any text outside of JSON.
"""

    try:
        ai_result_text = await analyze_with_image(prompt, contents)
        parsed_result = json.loads(ai_result_text)
    except Exception:
        parsed_result = {}

    try:
        imagekit_res = await upload_to_imagekit(contents, image.filename)
        image_url = (
            imagekit_res.get("image_url")
            if imagekit_res.get("status") == "ok"
            else None
        )
    except Exception:
        image_url = None

    data = {
        **parsed_result,
        "image_url": image_url or "No image available.",
    }

    result = await db.identifications.insert_one(data)
    data["_id"] = str(result.inserted_id)

    return data


@ai_router.get("/identifications")
@limiter.limit("45/minute")
async def get_all_identified_plant_species(
    request: Request,
):
    identifications = await db.identifications.find().to_list(length=100)
    for d in identifications:
        d["_id"] = str(d["_id"])
    return identifications


def build_prompt_from_sensor(
    sensor_data: dict, plant_name: str = "plant", plant_species: str = "species"
) -> str:
    """Constructs a detailed, structured prompt for the Gemini AI model."""
    prompt = f"""
Analyze the following plant sensor data and provide a health assessment.

**Context:**
- Plant Name: "{plant_name}"
- Plant Species: "{plant_species}"
- Timestamp of data: {sensor_data.get('timestamp', 'N/A')}
- Sensor Data JSON: {json.dumps(sensor_data, indent=2, default=str)}

**Task:**
1.  **Diagnose Condition:** Classify the plant's primary condition from this list: "Healthy", "Needs Water", "Overwatered", "Nutrient Deficiency", "Pest/Disease Issue", "Environmental Stress (Light/Temp)", "Other".
2.  **Estimate Confidence:** Provide a confidence score for your diagnosis (from 0.0 to 1.0).
3.  **Recommend Actions:** List up to 3 concrete, prioritized actions the user should take.
4.  **Watering Advice:** Give a clear watering recommendation (e.g., "Water thoroughly now", "Check again in 2 days", "Allow soil to dry out").
5.  **Provide Notes:** Add any brief, helpful notes or observations.

**Output Format:**
Respond with a valid JSON object ONLY. Do not include any text before or after the JSON block.
The JSON object must have these exact keys: "diagnosis", "confidence", "actions", "watering_recommendation", "notes".
Example:
{{
  "diagnosis": "Needs Water",
  "confidence": 0.95,
  "actions": [
    "Water the plant with 500ml of water immediately.",
    "Ensure the pot has adequate drainage.",
    "Move the plant to a location with slightly less direct sunlight."
  ],
  "watering_recommendation": "Water thoroughly now",
  "notes": "The soil moisture is very low and the temperature is high, indicating significant water loss."
}}
"""
    return prompt.strip()


# Prompt builder for the chatbot function
def build_chat_prompt(
    question: str, sensor_data: dict, last_analysis: Optional[dict]
) -> str:
    """Constructs a conversational prompt for the plant doctor chatbot."""

    # Safely get the last diagnosis text
    last_diagnosis_text = "No previous analysis available."
    if last_analysis and last_analysis.get("ai_result_parsed"):
        last_diagnosis_text = json.dumps(last_analysis["ai_result_parsed"], indent=2)

    prompt = f"""
You are 'VanaDristi', an expert plant doctor AI. Your goal is to provide simple, clear, and actionable advice to a plant owner based on their question and the latest data.

**Your Persona:**
- Professional, friendly, and encouraging.
- You break down complex topics into easy-to-understand advice.
- You NEVER respond in JSON format. You always respond in plain, conversational text.

**Here is the context you must use:**

1.  **Latest Sensor Data:**
    ```json
    {json.dumps(sensor_data, indent=2, default=str)}
    ```

2.  **Your Own Last Automated Analysis of This Data:**
    ```json
    {last_diagnosis_text}
    ```

3.  **The User's Current Question:**
    "{question}"

**Your Task:**
Based on all the context above, answer the user's question directly. Provide a helpful, professional, and simple solution.
"""
    return prompt.strip()


@ai_router.get("/analysis")
@limiter.limit("40/minute")
async def trigger_ai_analysis(
    request: Request,
    plant_id: str,
    use_image: bool = True,
):
    """
    Triggers an AI analysis using the latest sensor data for a given plant.
    If an image URL exists, it will be downloaded and sent to the AI.
    """
    query = {"plant_id": plant_id}
    latest_sensor_data = await db.sensor_data.find_one(
        query, {"_id": 0}, sort=[("timestamp", DESCENDING)]
    )

    if not latest_sensor_data:
        raise HTTPException(
            status_code=404, detail="No sensor data available to analyze."
        )

    plant_name = latest_sensor_data.get("plant_name", "Unknown Plant")
    plant_species = latest_sensor_data.get("plant_species", "Unknown Species")
    prompt = build_prompt_from_sensor(
        latest_sensor_data, plant_name=plant_name, plant_species=plant_species
    )

    img_bytes = None
    if use_image and (image_url := latest_sensor_data.get("image_url")):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url, timeout=10.0)
                response.raise_for_status()
                img_bytes = response.content
        except httpx.RequestError as e:
            print(f"Warning: Could not download image from {image_url}. Error: {e}")

    try:
        if img_bytes:
            ai_result_text = await analyze_with_image(prompt, img_bytes)
        else:
            ai_result_text = await analyze_text(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI model call failed: {e}")

    try:
        parsed_result = json.loads(ai_result_text)
    except json.JSONDecodeError:
        parsed_result = {"raw_text": ai_result_text}

    chat_document = {
        "timestamp": datetime.utcnow(),
        "plant_id": latest_sensor_data.get("plant_id", "default"),
        "sensor_data": latest_sensor_data,
        "ai_result_raw": ai_result_text,
        "ai_result_parsed": parsed_result,
    }
    await db.chat_history.insert_one(chat_document)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "ai_result": parsed_result,
            "sensor_data_used": json.loads(json.dumps(latest_sensor_data, default=str)),
        },
    )


# Chatbot endpoint
@ai_router.post("/chat", tags=["AI Analysis"])
@limiter.limit("40/minute")
async def chat_with_ai(
    request: Request,
    query: ChatQuery,
):
    """
    Engage in a conversation with the AI Plant Doctor about your plant's health.
    """
    plant_id = query.plant_id
    sensor_query = {"plant_id": plant_id}
    latest_sensor_data = await db.sensor_data.find_one(
        sensor_query, {"_id": 0}, sort=[("timestamp", DESCENDING)]
    )
    if not latest_sensor_data:
        raise HTTPException(
            status_code=404,
            detail=f"No sensor data found for plant_id '{plant_id}'. Cannot start chat.",
        )

    history_query = {"plant_id": plant_id, "type": "analysis"}
    last_analysis = await db.chat_history.find_one(
        history_query, {"_id": 0}, sort=[("timestamp", DESCENDING)]
    )

    prompt = build_chat_prompt(query.question, latest_sensor_data, last_analysis)

    try:
        ai_response_text = await analyze_text(prompt, response_format="text")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI model call failed: {e}")

    chat_document = {
        "timestamp": datetime.utcnow(),
        "plant_id": plant_id,
        "type": "chat",
        "user_question": query.question,
        "ai_response": ai_response_text,
    }
    await db.chat_history.insert_one(chat_document)

    return {"response": ai_response_text}


@ai_router.get("/chat_history")
@limiter.limit("45/minute")
async def get_chat_history(
    request: Request,
    plant_id: str = Query(default=None),
    limit: int = 50,
):
    """Retrieves the history of AI analysis interactions."""

    if plant_id:
        # Fetch latest entries for the given plant
        cursor = (
            db.chat_history.find({"plant_id": plant_id}, {"_id": 0})
            .sort("timestamp", DESCENDING)
            .limit(limit)
        )
        items = await cursor.to_list(length=limit)
    else:
        # Fetch latest entry per plant
        pipeline = [
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$plant_id", "latest_entry": {"$first": "$$ROOT"}}},
            {"$replaceRoot": {"newRoot": "$latest_entry"}},
            {"$limit": limit},
        ]
        cursor = db.chat_history.aggregate(pipeline)
        items = [doc async for doc in cursor]

    return json.loads(json.dumps(items, default=str))


# ==============================================================================
# 8. REGISTER ROUTERS
# ==============================================================================
app.include_router(home_router)
app.include_router(plant_router)
app.include_router(sensor_router)
app.include_router(ai_router)


# ==============================================================================
# 9. MAIN EXECUTION BLOCK [For Local Testing]
# ==============================================================================
# if __name__ == "__main__":
#     print("Starting VanaDristi API server...")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000)
