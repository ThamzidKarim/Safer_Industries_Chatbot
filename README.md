# Safer Industries Chatbot

This project is a chatbot for Safer Industries intern onboarding, built with Chainlit and a Postgres backend, deployed via Render and Supabase.

---

## How to clone and run locally

1. **Clone the repo**
```bash
git clone https://github.com/ThamzidKarim/Safer_Industries_Chatbot.git
cd safer-industries-chatbot
```

2. **Set up Python environment**
```bash
python3 -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set environment variables**
Create a .env file in the root folder with these variables:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
CHAINLIT_AUTH_SECRET=your_chainlit_secret_here
DATABASE_URL=postgresql://username:password@host:port/dbname
```
5. **Run the app**
```bash
chainlit run app.py
```

## Editing the RAG Document (JSON)
The chatbot uses a RAG (Retrieval-Augmented Generation) document stored as a JSON file: **doc.json**.
This file contains onboarding info.
To update:
  - Modify the value of the text key.
  - Use \n for new lines.
  - Save and restart the app to apply changes.


## Credits

Big thanks to 3CodeCampfor their YouTube tutorial https://www.youtube.com/watch?v=ozq9fK9Pn-s, which served as the foundation for this chatbot project.



