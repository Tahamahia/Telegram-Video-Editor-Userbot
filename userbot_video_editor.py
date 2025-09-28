#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== Userbot Video Editor (Pyrogram) =====
# - /new_edit يبدأ جلسة
# - تبعت فيديوات بالترتيب (video أو document video)
# - لكل فيديو: تبي تقص؟ (نعم/لا) وتدخل start/end مرات متعددة
# - بعدين اسم الملف + نسبة الضغط
# - يدمج: intro.mp4 + [كل المقاطع بعد القص] + يركّب frame.png على المقاطع فقط (مش على الانترو)
# - توحيد الدقة/FPS والصوت، ودعم كل الأبعاد (طولي/مربع/عرضي)
# - Fallback: h264_nvenc -> h264_qsv -> h264_amf -> libx264
# - قصّ متوازٍ، تنظيف ملفات مؤقتة

import os, json, shutil, asyncio, subprocess, concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from pyrogram import Client, filters
from pyrogram.types import Message

# ---------- إعدادات الحساب ----------
API_ID = ********
API_HASH = "**************************"
SESSION_NAME = "*************"  # ملف الجلسة

# ---------- ملفات ثابتة ----------
INTRO = Path("intro.mp4")  # الانترو بجانب السكربت
FRAME = Path("frame.png")  # الإطار بجانب السكربت
WORK_ROOT = Path("work")
MAX_WORKERS = max(2, (os.cpu_count() or 4) // 2)

# ---------- حالة الجلسة ----------
class Session:
    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.inputs: List[Path] = []
        self.cuts: Dict[str, List[Tuple[float,float]]] = {}
        self.awaiting_video = False
        self.awaiting_more_videos = False
        self.awaiting_cut_decision_for: Optional[Path] = None
        self.awaiting_cut_start: Optional[Path] = None
        self.awaiting_cut_end: Optional[Tuple[Path, float]] = None
        self.awaiting_final_name = False
        self.awaiting_compression = False
        self.final_name: Optional[str] = None
        self.compression_pct: Optional[int] = None
        self.workdir: Path = WORK_ROOT / str(chat_id)

SESSIONS: Dict[int, Session] = {}
BUSY_LOCK = asyncio.Lock()

def reset_session(chat_id: int):
    SESSIONS[chat_id] = Session(chat_id)

# ---------- أدوات FFmpeg ----------
def run(cmd: List[str]):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def try_run(cmd: List[str]) -> bool:
    print(">>", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def out_text(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True, errors="replace")

def ffprobe_json(path: Path) -> dict:
    out = out_text([
        "ffprobe","-v","error","-print_format","json",
        "-show_streams","-show_format",str(path)
    ])
    return json.loads(out)

def has_audio(path: Path) -> bool:
    d = ffprobe_json(path)
    return any(s.get("codec_type")=="audio" for s in d.get("streams",[]))

def get_resolution(path: Path) -> Tuple[int,int]:
    d = ffprobe_json(path)
    v = next((s for s in d.get("streams",[]) if s.get("codec_type")=="video"), None)
    if not v: return (1920,1080)
    return int(v.get("width",1920)), int(v.get("height",1080))

def get_fps(path: Path) -> int:
    d = ffprobe_json(path)
    v = next((s for s in d.get("streams",[]) if s.get("codec_type")=="video"), None)
    fr = v.get("avg_frame_rate") or (v.get("r_frame_rate") if v else None)
    if fr and fr != "0/0":
        try:
            num, den = fr.split("/")
            num = float(num); den = float(den)
            if den != 0:
                return max(1, round(num/den))
        except:
            pass
    return 30

def get_duration(path: Path) -> float:
    out = out_text([
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",
        str(path)
    ]).strip()
    return float(out or 0.0)

def ffmpeg_has_encoder(name: str) -> bool:
    try: return name in out_text(["ffmpeg","-hide_banner","-encoders"])
    except: return False

def encoders_order() -> List[str]:
    order=[]
    for e in ["h264_nvenc","h264_qsv","h264_amf"]:
        if ffmpeg_has_encoder(e): order.append(e)
    order.append("libx264")
    return order

def encoder_args(enc: str, quality: int, final: bool, threads: int) -> List[str]:
    if enc=="h264_nvenc":
        return ["-c:v","h264_nvenc","-preset","p6" if not final else "p5","-cq",str(quality),"-b:v","0","-pix_fmt","yuv420p"]
    if enc=="h264_qsv":
        return ["-c:v","h264_qsv","-global_quality",str(quality),"-preset","fast" if not final else "medium","-pix_fmt","nv12"]
    if enc=="h264_amf":
        return ["-c:v","h264_amf","-quality","balanced" if not final else "quality","-pix_fmt","yuv420p"]
    # CPU
    return ["-c:v","libx264","-crf",str(quality),"-preset","fast" if not final else "slow","-pix_fmt","yuv420p","-threads",str(os.cpu_count() or 8)]

def try_encode_with_fallback(build_cmd_fn, enc_list: List[str]) -> Tuple[bool, Optional[str]]:
    for enc in enc_list:
        cmd = build_cmd_fn(enc)
        if try_run(cmd):
            return True, enc
    return False, None

def seconds_from_str(t: str) -> Optional[float]:
    p = t.strip().split(":")
    try:
        if len(p)==1:  return float(p[0])
        if len(p)==2:  return int(p[0])*60+float(p[1])
        if len(p)==3:  return int(p[0])*3600+int(p[1])*60+float(p[2])
    except: return None
    return None

def quality_from_percent(pct: int, base_best: int = 18, base_worst: int = 28) -> int:
    pct = max(0, min(100, pct))
    q = base_best + (base_worst - base_best) * (pct/100.0)
    return int(round(q))

def cut_one_with_fallback(src: Path, keep: List[Tuple[float,float]], enc_list: List[str], quality:int, threads:int, outdir: Path) -> Path:
    # بدون قص: إعادة ترميز كاملة لضبط الصوت/الفيديو
    if keep==[(0, get_duration(src))]:
        outp = outdir/f"__part_{src.stem}_all.mp4"
        def build_cmd(enc):
            base=["ffmpeg","-y","-hwaccel","auto","-i",str(src)]
            base+=encoder_args(enc,quality,final=False,threads=threads)
            base+=["-c:a","aac","-b:a","160k","-movflags","+faststart",str(outp)]
            return base
        ok,_=try_encode_with_fallback(build_cmd, enc_list)
        if not ok: raise RuntimeError(f"All encoders failed for {src.name}")
        return outp

    # قص الأجزاء مع ترميز
    parts=[]
    for i,(s,e) in enumerate(keep):
        part = outdir/f"__part_{src.stem}_{i}.mp4"
        def build_cmd(enc):
            base=["ffmpeg","-y","-hwaccel","auto","-ss",str(s),"-to",str(e),"-i",str(src)]
            base+=encoder_args(enc,quality,final=False,threads=threads)
            base+=["-c:a","aac","-b:a","160k","-movflags","+faststart",str(part)]
            return base
        ok,_=try_encode_with_fallback(build_cmd, enc_list)
        if not ok:
            raise RuntimeError(f"All encoders failed for {src.name} part {i}")
        parts.append(part)

    if len(parts)==1:
        return parts[0]

    # دمج الأجزاء مؤقتًا (copy)
    txt = outdir/f"__concat_{src.stem}.txt"
    with open(txt,"w",encoding="utf-8") as f:
        for p in parts: f.write(f"file '{p.resolve().as_posix()}'\n")
    out = outdir/f"__trimmed_{src.stem}.mp4"
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(txt),"-c","copy","-movflags","+faststart",str(out)])
    return out

def process_all(inputs: List[Tuple[Path,List[Tuple[float,float]]]], final_name: str, percent_compression: int, workdir: Path) -> Path:
    if not INTRO.exists() or not FRAME.exists():
        raise RuntimeError("intro.mp4 or frame.png not found next to the script.")

    out_dir = workdir/"out"
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_list = encoders_order()
    threads  = os.cpu_count() or 8
    quality = quality_from_percent(percent_compression, base_best=18, base_worst=28)

    processed = [None] * len(inputs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, threads)) as exe:
        fut_map = {}
        for idx, (src, keep) in enumerate(inputs):
            fut = exe.submit(cut_one_with_fallback, src, keep, enc_list, quality, threads, out_dir)
            fut_map[fut] = idx
        for fut in concurrent.futures.as_completed(fut_map):
            processed[fut_map[fut]] = fut.result()

    # اعتمد أول ملف كنموذج للأبعاد و الـ FPS
    W,H  = get_resolution(processed[0])
    fps  = get_fps(processed[0]) or 30

    # بناء الأمر
    cmd = ["ffmpeg","-y","-hwaccel","auto","-i",str(INTRO)]
    for p in processed: cmd += ["-i",str(p)]
    cmd += ["-i",str(FRAME)]

    f = []
    # 1) الانترو بدون إطار
    f.append(f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p[v0i]")
    if has_audio(INTRO):
        f.append("[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0i]")
    else:
        f.append("anullsrc=r=48000:cl=stereo[a0i]")

    # 2) بقية المقاطع: توحيد + إطار
    clip_vs, clip_as = [], []
    for idx in range(1, 1+len(processed)):
        f.append(f"[{idx}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                 f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p[v{idx}]")
        f.append(f"[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]")
        clip_vs.append(f"[v{idx}]"); clip_as.append(f"[a{idx}]")

    # concat لكل المقاطع (بدون الانترو) -> vclips/aclips
    n_clips = len(processed)
    f.append("".join(clip_vs+clip_as)+f"concat=n={n_clips}:v=1:a=1[vclips][aclips]")

    # إطار فقط على المقاطع
    frame_idx = 1 + len(processed)
    f.append(f"[{frame_idx}:v]scale={W}:{H}[fr]")
    f.append(f"[vclips][fr]overlay=0:0:format=auto[vclips_out]")

    # 3) اجمع الانترو + المقاطع المأطرة
    f.append(f"[v0i][a0i][vclips_out][aclips]concat=n=2:v=1:a=1[cv][ca]")

    filter_complex = ";".join(f)

    def build_final(enc: str):
        base = cmd + ["-filter_complex", filter_complex, "-map","[cv]","-map","[ca]"]
        base += encoder_args(enc, quality, final=True, threads=threads)
        base += ["-movflags","+faststart", str(out_dir/f"{final_name}.mp4")]
        return base

    ok, used = try_encode_with_fallback(build_final, enc_list)
    if not ok:
        raise RuntimeError("All encoders failed on final render.")
    print(f"[Final] used encoder: {used}")
    return out_dir/f"{final_name}.mp4"

def build_keep_segments(cuts: List[Tuple[float,float]], dur: float) -> List[Tuple[float,float]]:
    segs=[]; prev=0.0
    for s,e in sorted(cuts):
        if s>prev: segs.append((prev,s))
        prev=max(prev,e)
    if prev<dur: segs.append((prev,dur))
    return segs

# ---------- Pyrogram app ----------
app = Client(SESSION_NAME, api_id=API_ID, api_hash=API_HASH, in_memory=False)

# فلتر فيديو يشمل video أو document بنوع video/*
VIDEO_FILTER = (filters.video |
                (filters.document & filters.create(lambda _, __, m: m.document and (m.document.mime_type or "").startswith("video/"))))

# /new_edit
@app.on_message(filters.private & filters.command("new_edit"))
async def new_edit_handler(client: Client, m: Message):
    if BUSY_LOCK.locked():
        await m.reply_text("⚠️ فيه مشروع قاعد يخدم توا. استنى شوية وجرب /new_edit.")
        return

    chat_id = m.chat.id
    reset_session(chat_id)

    sess = SESSIONS[chat_id]
    if sess.workdir.exists():
        shutil.rmtree(sess.workdir, ignore_errors=True)
    (sess.workdir/"in").mkdir(parents=True, exist_ok=True)
    (sess.workdir/"out").mkdir(parents=True, exist_ok=True)

    if not INTRO.exists() or not FRAME.exists():
        await m.reply_text("❌ لازم intro.mp4 و frame.png يكونوا جنب السكربت.")
        return

    sess.awaiting_video = True
    await m.reply_text("🆕 جلسة جديدة… ابعت **أول فيديو**.")

# استقبال فيديو/وثيقة فيديو
@app.on_message(filters.private & VIDEO_FILTER)
async def handle_video(client: Client, m: Message):
    chat_id = m.chat.id
    if chat_id not in SESSIONS:
        await m.reply_text("ابدا بـ /new_edit.")
        return

    sess = SESSIONS[chat_id]
    if not sess.awaiting_video:
        await m.reply_text("مش متوقع فيديو توا. لو تبي تبدا من جديد: /new_edit")
        return

    media = m.video or m.document
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        await m.reply_text("❌ هذا ملف Document مش فيديو.")
        return

    stem_index = len(sess.inputs) + 1
    local_path = sess.workdir/"in"/f"{stem_index}.mp4"

    await m.reply_text("⬇️ جاري التنزيل…")
    await client.download_media(m, file_name=str(local_path))

    sess.inputs.append(local_path)
    sess.cuts[local_path.stem] = []
    sess.awaiting_video = False
    sess.awaiting_cut_decision_for = local_path

    await m.reply_text(
        f"📥 تم استلام `{local_path.name}` (عدد الملفات: {len(sess.inputs)}).\n"
        f"تبي تقص منه أجزاء؟ (yes/no)"
    )

# نصوص التفاعل
@app.on_message(filters.private & filters.text & ~filters.command(["new_edit"]))
async def handle_text(client: Client, m: Message):
    chat_id = m.chat.id
    text_raw = m.text or ""
    text = text_raw.strip().lower()

    if BUSY_LOCK.locked():
        await m.reply_text("⚠️ فيه مشروع قاعد يخدم توا. استنى لين يكمل.")
        return

    if chat_id not in SESSIONS:
        await m.reply_text("ابدا بـ /new_edit.")
        return

    sess = SESSIONS[chat_id]

    # قرار القص
    if sess.awaiting_cut_decision_for is not None:
        if text in {"yes","y"}:
            sess.awaiting_cut_start = sess.awaiting_cut_decision_for
            sess.awaiting_cut_decision_for = None
            await m.reply_text("Start time (مثال 0:17):")
            return
        elif text in {"no","n"}:
            sess.awaiting_cut_decision_for = None
            sess.awaiting_more_videos = True
            await m.reply_text("تبي تضيف فيديو ثاني؟ (yes/no)")
            return
        else:
            await m.reply_text("جاوب yes أو no.")
            return

    # start time
    if sess.awaiting_cut_start is not None and sess.awaiting_cut_end is None:
        s = seconds_from_str(text_raw.strip())
        if s is None:
            await m.reply_text("❌ وقت غير صحيح. مثال: 0:17")
            return
        sess.awaiting_cut_end = (sess.awaiting_cut_start, s)
        await m.reply_text("End time (مثال 0:22):")
        return

    # end time
    if sess.awaiting_cut_end is not None:
        src, s = sess.awaiting_cut_end
        e = seconds_from_str(text_raw.strip())
        if e is None or e <= s:
            await m.reply_text("❌ End لازم يكون أكبر من Start. جرّب تاني.")
            return
        dur = get_duration(src)
        if s >= dur:
            await m.reply_text("❌ Start خارج مدة الفيديو.")
            sess.awaiting_cut_end = None
            return
        if e > dur: e = dur
        sess.cuts[src.stem].append((s,e))
        sess.awaiting_cut_end = None
        await m.reply_text("تبي تضيف قصّة ثانية؟ (yes/no)")
        sess.awaiting_cut_decision_for = src
        sess.awaiting_cut_start = None
        return

    # إضافة فيديوات أخرى؟
    if sess.awaiting_more_videos:
        if text in {"yes","y"}:
            sess.awaiting_more_videos = False
            sess.awaiting_video = True
            await m.reply_text("ابعث الفيديو التالي.")
            return
        elif text in {"no","n"}:
            sess.awaiting_more_videos = False
            sess.awaiting_final_name = True
            await m.reply_text("أوك. شن اسم الملف النهائي (بدون .mp4)؟")
            return
        else:
            await m.reply_text("جاوب yes أو no.")
            return

    # اسم الملف النهائي
    if sess.awaiting_final_name:
        name = text_raw.strip()
        if not name:
            await m.reply_text("أدخل اسم صحيح.")
            return
        sess.final_name = name
        sess.awaiting_final_name = False
        sess.awaiting_compression = True
        await m.reply_text(
            "نسبة الضغط (0% أعلى جودة، 100% أصغر حجم).\n"
            "أمثلة: 0، 10، 20، 30 …\n"
            "أرسل رقم بين 0 و 100:"
        )
        return

    # نسبة الضغط
    if sess.awaiting_compression:
        try:
            pct = int(text_raw.strip().replace("%",""))
        except:
            await m.reply_text("أرسل رقم 0..100 (مثال 20).")
            return
        if pct < 0 or pct > 100:
            await m.reply_text("أرسل رقم بين 0 و 100.")
            return
        sess.compression_pct = pct
        sess.awaiting_compression = False

        await m.reply_text("⏳ شغال على الفيديو…")
        asyncio.create_task(run_pipeline_and_respond(client, m, sess))
        return

    if sess.awaiting_video:
        await m.reply_text("ابعث الفيديو توا.")
    else:
        await m.reply_text("مش فاهم طلبك. تقدر تبدأ من جديد بـ /new_edit")

async def run_pipeline_and_respond(client: Client, m: Message, sess: Session):
    async with BUSY_LOCK:
        try:
            pairs=[]
            for p in sess.inputs:
                keep = build_keep_segments(sess.cuts[p.stem], get_duration(p)) if sess.cuts[p.stem] else [(0, get_duration(p))]
                pairs.append((p, keep))

            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(None, process_all, pairs, sess.final_name, sess.compression_pct, sess.workdir)

            await m.reply_text("✅ تم! جاري الرفع …")
            await client.send_video(
                chat_id=sess.chat_id,
                video=str(result_path),
                caption=f"{sess.final_name}.mp4"
            )
        except Exception as e:
            await m.reply_text(f"❌ Failed: {e}")
        finally:
            try:
                shutil.rmtree(sess.workdir, ignore_errors=True)
            except: pass
            reset_session(sess.chat_id)
            await client.send_message(chat_id=sess.chat_id, text="🧹 تم التنظيف. تقدر تبدأ /new_edit من جديد.")

# ---------- تشغيل ----------
if __name__ == "__main__":
    print("👤 Userbot starting… (Pyrogram)")
    app.run()
