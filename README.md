<div align="center">
  <br />
  <img src="https://placehold.co/600x300/1D2B1F/90EE90?text=VanaDristi+API&font=raleway" alt="VanaDristi API Banner">
  <br /><br />
  <h1>VanaDristi API 🌿</h1>
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

[cite_start]**VanaDristi** (from Sanskrit: वन दृष्टि, meaning "Forest Vision") is a public platform for monitoring plant health[cite: 1]. [cite_start]It leverages real-time sensor data and the power of Google's Gemini AI to provide detailed health assessments, actionable advice, and timely alerts via Telegram[cite: 1], acting as your personal AI botanist.

## ✨ Core Features

- 🤖 **AI-Powered Analysis**:
  - [cite_start]**Species Identification**: Identify a plant's name from an image[cite: 1].
  - [cite_start]**Health Diagnosis**: Get a detailed health report, confidence score, and actionable recommendations from sensor data[cite: 1].
  - [cite_start]**Conversational AI Chat**: Ask an AI plant doctor specific questions about your plant's condition[cite: 1].
- [cite_start]📊 **Real-time Sensor Monitoring**: A robust endpoint for ingesting data like soil moisture, temperature, humidity, and light intensity[cite: 1].
- [cite_start]🔔 **Automated Telegram Alerts**: Set custom thresholds for sensor data and receive instant notifications when your plant needs attention[cite: 1].
- [cite_start]📈 **Historical Data & Trends**: Access the latest sensor data, a complete history of readings, and aggregated daily trends[cite: 1].
- [cite_start]🌱 **Full Plant Management**: Complete CRUD (Create, Read, Update, Delete) functionality for managing your plant profiles[cite: 1].
- [cite_start]🚀 **Scalable & Secure**: Built with modern, asynchronous tools (FastAPI & Motor), with rate limiting to prevent abuse[cite: 1].
- ☁️ **Vercel Ready**: Pre-configured for seamless, serverless deployment.

## 🏗️ System Architecture

The API follows a simple yet powerful data flow:

1.  [cite_start]**Device/User**: A sensor device (like an ESP32) or a user uploads sensor data and an optional image to the `/sensor/upload` endpoint[cite: 1].
2.  **FastAPI Backend**: The data is received, validated, and stored in MongoDB. [cite_start]The image is uploaded to ImageKit[cite: 1].
3.  [cite_start]**Background Task**: A background task checks the new data against the plant's alert thresholds[cite: 1].
4.  [cite_start]**Telegram Alert**: If a threshold is breached, a formatted alert is sent to a specified Telegram chat[cite: 1].
5.  [cite_start]**AI Analysis**: The user can trigger an AI analysis, which pulls the latest data from MongoDB and sends it to the Google Gemini API[cite: 1].
6.  [cite_start]**AI Response**: The AI's diagnosis or chat response is saved to the database and returned to the user[cite: 1].

<div align="center">

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fsnehkr%2FVanaDristi-Backend)

</div>

## 🛠️ Tech Stack

| Category       | Technology                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Backend**    | [**FastAPI**](https://fastapi.tiangolo.com/), [**Uvicorn**](https://www.uvicorn.org/)                                                      |
| **Database**   | [cite_start][**MongoDB**](https://www.mongodb.com/) (with [**Motor**](https://motor.readthedocs.io/en/stable/) for async access) [cite: 1] |
| **AI Model**   | [**Google Gemini**](https://ai.google.dev/)                                                                                                |
| **Image CDN**  | [cite_start][**ImageKit.io**](https://imagekit.io/) [cite: 1]                                                                              |
| **Alerting**   | [cite_start][**Telegram**](https://telegram.org/) [cite: 1]                                                                                |
| **Deployment** | [**Vercel**](https://vercel.com/)                                                                                                          |

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

[cite_start]Here is a summary of the available endpoints[cite: 1]. For full details, run the server and visit the `/docs` endpoint.

| Endpoint                                | Method   | Description                                                                 |
| --------------------------------------- | -------- | --------------------------------------------------------------------------- |
| **Plant Management (`/api/v1/plants`)** |          |                                                                             |
| `/`                                     | `POST`   | [cite_start]Create a new plant profile. [cite: 1]                           |
| `/`                                     | `GET`    | [cite_start]Get a list of all plants. [cite: 1]                             |
| `/{plant_id}`                           | `GET`    | [cite_start]Get details for a specific plant. [cite: 1]                     |
| `/{plant_id}`                           | `PUT`    | [cite_start]Update a plant's details and alert thresholds. [cite: 1]        |
| `/{plant_id}`                           | `DELETE` | [cite_start]Delete a plant. [cite: 1]                                       |
| **Sensor Data (`/api/v1/sensor`)**      |          |                                                                             |
| `/upload`                               | `POST`   | [cite_start]Upload sensor data (JSON) and an optional image file. [cite: 1] |
| `/history/{plant_id}`                   | `GET`    | [cite_start]Get the historical sensor data for a plant. [cite: 1]           |
| `/latest/{plant_id}`                    | `GET`    | [cite_start]Get the most recent sensor reading for a plant. [cite: 1]       |
| `/trends/{plant_id}`                    | `GET`    | [cite_start]Get daily average trends for key metrics. [cite: 1]             |
| **AI Analysis (`/api/v1/ai`)**          |          |                                                                             |
| `/identify`                             | `POST`   | [cite_start]Upload an image to identify the plant species. [cite: 1]        |
| `/analysis`                             | `GET`    | [cite_start]Trigger an AI health analysis for a plant. [cite: 1]            |
| `/chat`                                 | `POST`   | [cite_start]Start a conversation with the AI plant doctor. [cite: 1]        |
| `/chat_history`                         | `GET`    | [cite_start]Retrieve the history of AI interactions for a plant. [cite: 1]  |

---

<div align="center">
  Made with ❤️ by Sneh Kumar
</div>
