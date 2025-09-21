<div align="center">
  <br />
  <img src="https://ik.imagekit.io/vanadristi/VanaDristi.jpeg" alt="VanaDristi API Banner" width="250px" style="border-radius: 50%;">
  <br />
  <h1>🌿 VanaDristi API 🌿</h1>
  <p>
    <b>An intelligent, asynchronous API for AI-powered plant health monitoring, diagnosis, and care.</b>
  </p>
  <br />

  <a href="https://github.com/snehkr/VanaDristi-Backend">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  </a>
  <a href="https://www.mongodb.com/">
    <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB">
  </a>
  <a href="https://ai.google.dev/">
    <img src="https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google Gemini">
  </a>
  <a href="https://github.com/snehkr/VanaDristi-Backend/stargazers">
    <img src="https://img.shields.io/github/stars/snehkr/VanaDristi-Backend?style=for-the-badge&color=ffd000" alt="GitHub stars">
  </a>
  <a href="https://github.com/snehkr/VanaDristi-Backend/network/members">
    <img src="https://img.shields.io/github/forks/snehkr/VanaDristi-Backend?style=for-the-badge&color=blueviolet" alt="GitHub forks">
  </a>

</div>

---

**VanaDristi** (from Sanskrit: वन दृष्टि, meaning "Forest Vision") is a public platform for monitoring plant health. It leverages real-time sensor data and the power of Google's Gemini AI to provide detailed health assessments, actionable advice, and timely alerts via Telegram, acting as your personal AI botanist.

## ✨ Core Features

- 🤖 **AI-Powered Analysis**:
  - **Species Identification**: Identify a plant's name from an image.
  - **Health Diagnosis**: Get a detailed health report, confidence score, and actionable recommendations from sensor data.
  - **Conversational AI Chat**: Ask an AI plant doctor specific questions about your plant's condition.
- 📊 **Real-time Sensor Monitoring**: A robust endpoint for ingesting data like soil moisture, temperature, humidity, and light intensity.
- 🔔 **Automated Telegram Alerts**: Set custom thresholds for sensor data and receive instant notifications when your plant needs attention.
- 📈 **Historical Data & Trends**: Access the latest sensor data, a complete history of readings, and aggregated daily trends.
- 🌱 **Full Plant Management**: Complete CRUD (Create, Read, Update, Delete) functionality for managing your plant profiles.
- 🚀 **Scalable & Secure**: Built with modern, asynchronous tools (FastAPI & Motor), with rate limiting to prevent abuse.
- ☁️ **Vercel Ready**: Pre-configured for seamless, serverless deployment.

## 🏗️ System Architecture

The API follows a simple yet powerful data flow:

1.  **Device/User**: A sensor device (like an ESP32) or a user uploads sensor data and an optional image to the `/sensor/upload` endpoint.
2.  **FastAPI Backend**: The data is received, validated, and stored in MongoDB. The image is uploaded to ImageKit.
3.  **Background Task**: A background task checks the new data against the plant's alert thresholds.
4.  **Telegram Alert**: If a threshold is breached, a formatted alert is sent to a specified Telegram chat.
5.  **AI Analysis**: The user can trigger an AI analysis, which pulls the latest data from MongoDB and sends it to the Google Gemini API.
6.  **AI Response**: The AI's diagnosis or chat response is saved to the database and returned to the user.

<div align="center">

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fsnehkr%2FVanaDristi-Backend)

</div>

## 🛠️ Tech Stack

| Category       | Technology                                                                                                           |
| -------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Backend**    | [**FastAPI**](https://fastapi.tiangolo.com/), [**Uvicorn**](https://www.uvicorn.org/)                                |
| **Database**   | [**MongoDB**](https://www.mongodb.com/) (with [**Motor**](https://motor.readthedocs.io/en/stable/) for async access) |
| **AI Model**   | [**Google Gemini**](https://ai.google.dev/)                                                                          |
| **Image CDN**  | [**ImageKit.io**](https://imagekit.io/)                                                                              |
| **Alerting**   | [**Telegram**](https://telegram.org/)                                                                                |
| **Deployment** | [**Vercel**](https://vercel.com/)                                                                                    |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A MongoDB instance (local or on [Atlas](https://www.mongodb.com/cloud/atlas))
- API keys for Google Gemini, ImageKit.io, and a Telegram Bot.

### Local Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/snehkr/VanaDristi-Backend.git](https://github.com/snehkr/VanaDristi-Backend.git)
    cd VanaDristi-Backend
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root. Use the template below and fill in your credentials.

    ```env
    # MongoDB Configuration
    MONGODB_URL="your_mongodb_connection_string"
    DB_NAME="vana_dristi_ai"

    # Gemini AI Configuration
    GEMINI_API_KEY="your_gemini_api_key"

    # ImageKit.io Configuration
    IMAGEKIT_URL="[https://ik.imagekit.io/your_instance](https://ik.imagekit.io/your_instance)"
    IMAGEKIT_PRIVATE_KEY="your_imagekit_private_key"
    IMAGEKIT_UPLOAD_FOLDER="VanaDristi"

    # Telegram Notification Configuration
    TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
    TELEGRAM_CHAT_ID="your_telegram_chat_id"
    ```

5.  **Run the Application:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be live at `http://127.0.0.1:8000/docs` for the interactive documentation.

## 📖 API Endpoint Documentation

Here is a summary of the available endpoints. For full details, run the server and visit the `/docs` endpoint.

| Endpoint                                | Method   | Description                                           |
| --------------------------------------- | -------- | ----------------------------------------------------- |
| **Plant Management (`/api/v1/plants`)** |          |                                                       |
| `/`                                     | `POST`   | Create a new plant profile.                           |
| `/`                                     | `GET`    | Get a list of all plants.                             |
| `/{plant_id}`                           | `GET`    | Get details for a specific plant.                     |
| `/{plant_id}`                           | `PUT`    | Update a plant's details and alert thresholds.        |
| `/{plant_id}`                           | `DELETE` | Delete a plant.                                       |
| `/latest`                               | `GET`    | Get Latest Plant ID for observation                   |
| `/latest`                               | `POST`   | Set Latest Plant ID for observation                   |
| `/latest`                               | `DELETE` | Delete Latest Plant ID for observation                |
| **Sensor Data (`/api/v1/sensor`)**      |          |                                                       |
| `/upload`                               | `POST`   | Upload sensor data (JSON) and an optional image file. |
| `/upload_first`                         | `POST`   | Upload sensor data (JSON) and an image file with AI.  |
| `/history/{plant_id}`                   | `GET`    | Get the historical sensor data for a plant.           |
| `/latest/{plant_id}`                    | `GET`    | Get the most recent sensor reading for a plant.       |
| `/trends/{plant_id}`                    | `GET`    | Get daily average trends for key metrics.             |
| **AI Analysis (`/api/v1/ai`)**          |          |                                                       |
| `/identify`                             | `POST`   | Upload an image to identify the plant species.        |
| `/identifications`                      | `GET`    | Get All Identified Plant Species as Json.             |
| `/analysis`                             | `GET`    | Trigger an AI health analysis for a plant.            |
| `/chat`                                 | `POST`   | Start a conversation with the AI plant doctor.        |
| `/chat_history`                         | `GET`    | Retrieve the history of AI interactions for a plant.  |

---

<div align="center">
  Made with ❤️ by Sneh Kumar
</div>
