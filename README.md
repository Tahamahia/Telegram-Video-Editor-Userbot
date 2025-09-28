# 🖥️ Telegram Video Editor Userbot

This is a **Userbot** built with Python using [Pyrogram](https://docs.pyrogram.org/).  
It receives videos via Telegram (private chat), lets you cut them, merge multiple clips, and automatically add an intro (`intro.mp4`) and overlay frame (`frame.png`).

---

## ✨ Features
- Accepts both `video` and `document video` uploads.
- Multiple cuts per video.
- Merge videos with intro and frame.
- Normalize resolution, FPS, and audio (48kHz Stereo).
- Adjustable compression (0% = best quality, 100% = smallest size).
- Auto cleanup of temp files after sending.

---

## ⚙️ Requirements
- Python 3.9+
- Libraries: Pyrogram, TgCrypto, FFmpeg
- FFmpeg installed on the system (`sudo apt install ffmpeg`)

---

## 🚀 Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Telegram-Video-Editor-Userbot.git
   cd Telegram-Video-Editor-Userbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set environment variables (API_ID, API_HASH, SESSION_NAME):
   ```bash
   export API_ID=12345
   export API_HASH="your_api_hash"
   export SESSION_NAME="ofuq_editor_userbot"
   ```

4. Place your files:
   - `intro.mp4` (intro video)
   - `frame.png` (overlay frame)

5. Run the bot:
   ```bash
   python userbot_video_editor.py
   ```

---

## 🔒 Security Notes
- **Never commit** your `.session` file to GitHub.
- Do not hardcode your API_ID and API_HASH in the script. Use environment variables.

---

# 🖥️ محرر الفيديو بتليجرام (Userbot)

هذا مشروع **يوزربوت** بلغة Python باستخدام مكتبة [Pyrogram](https://docs.pyrogram.org/).  
يقوم باستقبال الفيديوهات عبر التليجرام (رسائل خاصة)، ثم يسمح لك بقصها ودمجها وإضافة انترو (intro.mp4) وإطار (frame.png) تلقائيًا.

---

## ✨ المميزات
- استلام فيديوهات (video أو document video).
- قصّ مقاطع متعددة.
- دمج الفيديوهات مع الانترو والإطار.
- ضبط الدقة، الصوت (48kHz Stereo) و FPS.
- ضغط الفيديو بنسبة من 0% (أفضل جودة) إلى 100% (أصغر حجم).
- تنظيف تلقائي بعد الإرسال.

---

## ⚙️ المتطلبات
- Python 3.9+
- مكتبات: Pyrogram, TgCrypto, FFmpeg
- تثبيت FFmpeg على النظام (`sudo apt install ffmpeg`)

---

## 🚀 طريقة التشغيل
1. استنسخ المشروع:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Telegram-Video-Editor-Userbot.git
   cd Telegram-Video-Editor-Userbot
   ```

2. أنشئ بيئة افتراضية وثبّت المتطلبات:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. أنشئ متغيرات البيئة (API_ID, API_HASH, SESSION_NAME):
   ```bash
   export API_ID=12345
   export API_HASH="your_api_hash"
   export SESSION_NAME="ofuq_editor_userbot"
   ```

4. ضع ملفاتك:
   - `intro.mp4` (الفيديو الافتتاحي)
   - `frame.png` (الإطار الشفاف فوق الفيديو)

5. شغّل البوت:
   ```bash
   python userbot_video_editor.py
   ```

---

## 🔒 الملاحظات الأمنية
- **لا ترفع** ملف الجلسة (`.session`) إلى GitHub.
- لا تضع الـ API_ID و API_HASH في الكود. استعمل متغيرات بيئة.

---
