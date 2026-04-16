# Smart FAQ Chatbot (RAG)

Streamlit asosidagi FAQ chatbot: hujjatlarni yuklaydi, embedding yaratadi, semantik qidiruv qiladi va LLM orqali javob beradi.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

`.streamlit/secrets.toml` ichiga:

```toml
OPENAI_API_KEY="your_api_key"
APP_PASSWORD="user_access_password"   # optional
ADMIN_PASSWORD="admin_upload_password" # optional
```

## Streamlit Cloud Deploy

1. Loyihani GitHub repoga push qiling.
2. https://share.streamlit.io orqali repo tanlab deploy qiling.
3. `Settings -> Secrets` bo'limiga `OPENAI_API_KEY` qo'shing.
   ixtiyoriy:
   - `APP_PASSWORD`: appga kirish paroli
   - `ADMIN_PASSWORD`: faq upload qilish uchun admin paroli
4. App file: `app.py`.

Deploy bo'lgandan keyin sizda live URL hosil bo'ladi:

`https://<your-app-name>.streamlit.app`

## Access Modes

- Faqat Streamlit Cloud orqali:
  - `Public`: URL bor har kim ochadi
  - `Private`: faqat workspace/memberlar
- App ichidagi parol himoyasi:
  - `APP_PASSWORD` bo'lsa, barcha user login qiladi
  - `ADMIN_PASSWORD` bo'lsa, upload faqat admin parol bilan kirgan userga ochiladi

## Safety Features

- Rate limit: `12 request/minute` (session bo'yicha).
- Upload validation:
  - faqat `.txt`, `.md`, `.json`, `.pdf`
  - max file size: `10 MB`
- Admin source manager:
  - yuklangan source'larni sidebar orqali tanlab o'chirish mumkin.
