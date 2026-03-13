# -*- coding: utf-8 -*-
"""
SenseVoice FastAPI 服务 - 支持 HTTP API 和 WebSocket
===============================================
基于本地 ONNX 模型的语音识别服务，支持:
1. HTTP API: POST /transcribe - 上传音频文件进行识别
2. HTTP API: POST /transcribe/url - 通过 URL 识别远程音频
3. WebSocket: /ws/transcribe - 实时流式识别

模型路径配置:
    MODEL_DIR = "C:\\Users\\lihs\\AppData\\Roaming\\Shandianshuo\\models\\sensevoice-small"

启动服务:
    python api_server.py
    # 或
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

接口文档:
    http://127.0.0.1:8000/docs
    http://127.0.0.1:8000/redoc
"""

import os
import re
import json
import base64
import asyncio
import numpy as np
import librosa
from io import BytesIO
from typing import Union, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
import aiohttp
import uvicorn

# ─── 配置区域 ───────────────────────────────────────────────────────────────
MODEL_DIR = r"C:\Users\lihs\AppData\Roaming\Shandianshuo\models\sensevoice-small"
QUANTIZE = False
DEVICE_ID = -1  # -1=CPU, 0=GPU
BATCH_SIZE = 1
DEFAULT_LANGUAGE = "auto"
# ─────────────────────────────────────────────────────────────────────────────

# 语言 ID 映射
LID_DICT = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}

# 文本规范化 ID 映射
TEXTNORM_DICT = {
    "withitn": 14,
    "woitn": 15,
}

# 特殊标签过滤
_LABEL_PATTERN = re.compile(r"<\|[^|]+\|>")


def _strip_labels(text: str) -> str:
    """去除文本中的所有 <|xxx|> 特殊标签。"""
    return _LABEL_PATTERN.sub("", text).strip()


# ─── Token 解码器 ───────────────────────────────────────────────────────────


class TokensJsonDecoder:
    """基于 tokens.json 的解码器。"""

    def __init__(self, tokens_path: str):
        with open(tokens_path, encoding="utf-8") as f:
            token_list = json.load(f)
        self.id2token: List[str] = token_list
        self.token2id = {t: i for i, t in enumerate(token_list)}

    def decode(self, token_ids: List[int]) -> str:
        """将 token id 列表还原为字符串。"""
        pieces = [self.id2token[i] for i in token_ids if 0 < i < len(self.id2token)]
        text = "".join(pieces)
        text = text.replace("▁", " ").strip()
        return text


# ─── 模型推理核心 ───────────────────────────────────────────────────────────


class SenseVoiceInfer:
    """SenseVoice Small ONNX 推理类。"""

    def __init__(
        self,
        model_dir: str,
        quantize: bool = False,
        device_id: int = -1,
        batch_size: int = 1,
    ):
        self._validate_dir(model_dir, quantize)

        # 加载词汇表
        tokens_path = os.path.join(model_dir, "tokens.json")
        self.decoder = TokensJsonDecoder(tokens_path)
        self.blank_id = 0

        # 初始化声学前端
        config_path = os.path.join(model_dir, "config.yaml")
        frontend_conf = self._read_frontend_conf(config_path)
        frontend_conf["cmvn_file"] = None

        try:
            from funasr_onnx.utils.frontend import WavFrontend
        except ImportError:
            raise ImportError(
                "需要 funasr_onnx 提供 WavFrontend 特征提取。\n"
                "请安装：pip install funasr_onnx"
            )
        self.frontend = WavFrontend(**frontend_conf)
        self.sample_rate: int = int(self.frontend.opts.frame_opts.samp_freq)

        # 初始化 ONNX 推理会话
        model_file = os.path.join(
            model_dir, "model_quant.onnx" if quantize else "model.onnx"
        )
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        providers = ["CPUExecutionProvider"]
        if device_id >= 0:
            providers = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(
            model_file, sess_options=sess_options, providers=providers
        )
        self.batch_size = batch_size

        print(
            f"[✓] 模型加载完成，采样率：{self.sample_rate} Hz，词表大小：{len(self.decoder.id2token)}"
        )

    @staticmethod
    def _validate_dir(model_dir: str, quantize: bool):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在：{model_dir}")
        model_file = "model_quant.onnx" if quantize else "model.onnx"
        for fname in [model_file, "tokens.json", "config.yaml"]:
            fp = os.path.join(model_dir, fname)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"缺少模型文件：{fp}")

    @staticmethod
    def _read_frontend_conf(config_path: str) -> dict:
        """从 config.yaml 中读取 frontend_conf 节。"""
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("frontend_conf", {})

    def _load_audio(self, audio: Union[str, np.ndarray, BytesIO]) -> np.ndarray:
        """将各种格式的音频输入统一转为 float32 numpy 数组（16kHz mono）。"""
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        if isinstance(audio, str):
            waveform, _ = librosa.load(audio, sr=self.sample_rate, mono=True)
            return waveform.astype(np.float32)
        if isinstance(audio, BytesIO):
            waveform, _ = librosa.load(audio, sr=self.sample_rate, mono=True)
            return waveform.astype(np.float32)
        raise TypeError(f"不支持的音频类型：{type(audio)}")

    def _extract_features(self, waveforms: List[np.ndarray]):
        """提取 LFR-Fbank 特征，返回 (feats, feats_len)。"""
        feats_list, feats_len_list = [], []
        for waveform in waveforms:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats_list.append(feat)
            feats_len_list.append(feat_len)

        max_len = max(feats_len_list)
        padded = []
        for feat in feats_list:
            pad_width = ((0, max_len - feat.shape[0]), (0, 0))
            padded.append(np.pad(feat, pad_width, "constant", constant_values=0))

        feats = np.array(padded, dtype=np.float32)
        feats_len = np.array(feats_len_list, dtype=np.int32)
        return feats, feats_len

    def _infer(
        self,
        feats: np.ndarray,
        feats_len: np.ndarray,
        language: np.ndarray,
        textnorm: np.ndarray,
    ):
        """调用 ONNX 模型推理。"""
        outputs = self.session.run(
            None,
            {
                "speech": feats,
                "speech_lengths": feats_len,
                "language": language,
                "textnorm": textnorm,
            },
        )
        return outputs[0], outputs[1]

    @staticmethod
    def _greedy_decode_logits(logits: np.ndarray, valid_len: int) -> List[int]:
        """CTC 贪心解码。"""
        x = logits[:valid_len]
        yseq = np.argmax(x, axis=-1)
        deduped = [yseq[0]]
        for tok in yseq[1:]:
            if tok != deduped[-1]:
                deduped.append(tok)
        return [t for t in deduped if t != 0]

    def __call__(
        self,
        audio: Union[str, np.ndarray, BytesIO, List[str]],
        language: str = "auto",
        use_itn: bool = True,
    ) -> List[str]:
        """
        推理主入口。

        参数
        ----
        audio    : 音频输入（文件路径、np.ndarray、BytesIO 或路径列表）
        language : 语言代码（auto/zh/en/ja/ko/yue/nospeech）
        use_itn  : 是否开启逆文本规范化

        返回
        ----
        List[str]：每条音频对应的识别文本
        """
        if language not in LID_DICT:
            raise ValueError(f"不支持的语言：{language}，可选：{list(LID_DICT.keys())}")

        lid = LID_DICT[language]
        tid = TEXTNORM_DICT["withitn" if use_itn else "woitn"]

        # 统一加载音频
        if isinstance(audio, list):
            waveforms = [self._load_audio(a) for a in audio]
        else:
            waveforms = [self._load_audio(audio)]

        results = []
        for beg in range(0, len(waveforms), self.batch_size):
            batch = waveforms[beg : beg + self.batch_size]
            feats, feats_len = self._extract_features(batch)

            B = feats.shape[0]
            lang_arr = np.array([lid] * B, dtype=np.int32)
            tnorm_arr = np.array([tid] * B, dtype=np.int32)

            ctc_logits, enc_out_lens = self._infer(
                feats, feats_len, lang_arr, tnorm_arr
            )

            for b in range(B):
                token_ids = self._greedy_decode_logits(
                    ctc_logits[b], int(enc_out_lens[b])
                )
                text = self.decoder.decode(token_ids)
                results.append(text)

        return results


# ─── 全局模型实例 ───────────────────────────────────────────────────────────

model: Optional[SenseVoiceInfer] = None


def get_model() -> SenseVoiceInfer:
    """获取或初始化模型实例（懒加载）。"""
    global model
    if model is None:
        print(f"[*] 正在加载模型：{MODEL_DIR}")
        print(f"[*] 模型文件：{'model_quant.onnx' if QUANTIZE else 'model.onnx'}")
        print(f"[*] 设备：{'CPU' if DEVICE_ID < 0 else f'GPU:{DEVICE_ID}'}")
        model = SenseVoiceInfer(
            model_dir=MODEL_DIR,
            quantize=QUANTIZE,
            device_id=DEVICE_ID,
            batch_size=BATCH_SIZE,
        )
    return model


def transcribe_audio(
    audio: Union[str, np.ndarray, BytesIO],
    language: str = DEFAULT_LANGUAGE,
    use_itn: bool = True,
) -> dict:
    """
    将音频转化为文字。

    返回
    ----
    dict:
        text        : 纯文本（已去除标签）
        label_text  : 原始输出（含标签）
        language    : 使用的语言参数
        duration    : 处理时长（秒）
    """
    import time

    start_time = time.time()
    m = get_model()
    raw_list = m(audio, language=language, use_itn=use_itn)
    raw_text = raw_list[0] if raw_list else ""
    clean_text = _strip_labels(raw_text)
    duration = time.time() - start_time

    return {
        "text": clean_text,
        "label_text": raw_text,
        "language": language,
        "duration": round(duration, 3),
    }


# ─── Pydantic 模型 ──────────────────────────────────────────────────────────


class TranscriptionResponse(BaseModel):
    """转写响应模型"""

    success: bool = Field(..., description="是否成功")
    text: str = Field(default="", description="识别结果（纯文本）")
    label_text: Optional[str] = Field(None, description="原始输出（含标签）")
    language: str = Field(default="", description="识别语言")
    duration: float = Field(default=0.0, description="处理时长（秒）")
    message: Optional[str] = Field(None, description="错误信息（失败时）")


class TranscriptionRequest(BaseModel):
    """转写请求模型（URL方式）"""

    url: HttpUrl = Field(..., description="音频文件URL")
    language: str = Field(default="auto", description="语言代码")
    use_itn: bool = Field(default=True, description="是否使用逆文本规范化")


class WSMessage(BaseModel):
    """WebSocket 消息模型"""

    action: str = Field(..., description="操作类型：transcribe")
    audio_base64: Optional[str] = Field(None, description="Base64编码的音频数据")
    language: Optional[str] = Field("auto", description="语言代码")
    use_itn: Optional[bool] = Field(True, description="是否使用ITN")


# ─── FastAPI 应用 ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时预加载模型
    print("=" * 60)
    print("SenseVoice FastAPI 服务启动中...")
    print("=" * 60)
    get_model()
    print("=" * 60)
    print(f"API 文档: http://127.0.0.1:8000/docs")
    print(f"WebSocket: ws://127.0.0.1:8000/ws/transcribe")
    print("=" * 60)
    yield
    # 关闭时清理资源
    print("[*] 服务关闭，清理资源...")


app = FastAPI(
    title="SenseVoice 语音识别服务",
    description="基于 ONNX 的 SenseVoice Small 模型，支持 HTTP API 和 WebSocket 调用",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS - 允许所有来源跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


# ─── HTTP API 接口 ──────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """根路径 - 服务状态检查"""
    return {
        "service": "SenseVoice ASR API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "transcribe_file": "POST /transcribe",
            "transcribe_url": "POST /transcribe/url",
            "websocket": "WS /ws/transcribe",
        },
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(..., description="音频文件（支持 wav/mp3/flac/ogg 等）"),
    language: str = Form(default="auto", description="语言代码：auto/zh/en/ja/ko/yue"),
    use_itn: bool = Form(default=True, description="是否开启逆文本规范化"),
    keep_labels: bool = Form(default=False, description="是否在响应中包含原始标签"),
):
    """
    上传音频文件进行语音识别。

    **支持格式**: wav, mp3, flac, ogg 等 librosa 支持的格式

    **语言代码**:
    - auto: 自动检测（默认）
    - zh: 中文
    - en: 英文
    - ja: 日语
    - ko: 韩语
    - yue: 粤语

    **示例**:
    ```bash
    curl -X POST "http://localhost:8000/transcribe" \\
        -F "file=@test.mp3" \\
        -F "language=zh"
    ```
    """
    # 验证文件
    filename = file.filename or ""
    content_type = file.content_type or ""
    if not (
        content_type.startswith("audio/")
        or filename.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传有效的音频文件（wav/mp3/flac/ogg）",
        )

    try:
        # 读取音频数据
        audio_bytes = await file.read()
        audio_io = BytesIO(audio_bytes)

        # 执行识别
        result = transcribe_audio(audio_io, language=language, use_itn=use_itn)

        return TranscriptionResponse(
            success=True,
            text=result["text"],
            label_text=result["label_text"] if keep_labels else None,
            language=result["language"],
            duration=result["duration"],
        )

    except Exception as e:
        return TranscriptionResponse(
            success=False,
            text="",
            label_text=None,
            language=language,
            duration=0.0,
            message=str(e),
        )


@app.post("/transcribe/url", response_model=TranscriptionResponse)
async def transcribe_url(
    request: TranscriptionRequest,
    keep_labels: bool = False,
):
    """
    通过 URL 识别远程音频文件。

    **示例**:
    ```bash
    curl -X POST "http://localhost:8000/transcribe/url" \\
        -H "Content-Type: application/json" \\
        -d '{"url": "https://example.com/audio.mp3", "language": "zh"}'
    ```
    """
    try:
        # 下载远程音频
        async with aiohttp.ClientSession() as session:
            async with session.get(
                str(request.url),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0.36"
                },
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"无法下载音频文件：HTTP {response.status}",
                    )
                audio_bytes = await response.read()

        # 执行识别
        audio_io = BytesIO(audio_bytes)
        result = transcribe_audio(
            audio_io, language=request.language, use_itn=request.use_itn
        )

        return TranscriptionResponse(
            success=True,
            text=result["text"],
            label_text=result["label_text"] if keep_labels else None,
            language=result["language"],
            duration=result["duration"],
        )

    except HTTPException:
        raise
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            text="",
            label_text=None,
            language=request.language,
            duration=0.0,
            message=str(e),
        )


# ─── WebSocket 接口 ─────────────────────────────────────────────────────────


class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] 新连接建立，当前连接数：{len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"[WS] 连接断开，当前连接数：{len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket 实时语音识别接口。

    **连接方式**:
    ```javascript
    const ws = new WebSocket("ws://localhost:8000/ws/transcribe");
    ```

    **请求消息格式**:
    ```json
    {
        "action": "transcribe",
        "audio_base64": "base64_encoded_audio_data...",
        "language": "zh",
        "use_itn": true
    }
    ```

    **响应消息格式**:
    ```json
    {
        "success": true,
        "text": "识别结果",
        "label_text": "<|zh|><|NEUTRAL|><|Speech|>识别结果",
        "language": "zh",
        "duration": 1.234
    }
    ```
    """
    await manager.connect(websocket)
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()

            # 解析请求
            action = data.get("action", "transcribe")
            audio_base64 = data.get("audio_base64")
            language = data.get("language", "auto")
            use_itn = data.get("use_itn", True)
            keep_labels = data.get("keep_labels", False)

            if action != "transcribe":
                await manager.send_message(
                    websocket,
                    {
                        "success": False,
                        "message": f"不支持的操作：{action}",
                    },
                )
                continue

            if not audio_base64:
                await manager.send_message(
                    websocket,
                    {"success": False, "message": "缺少 audio_base64 字段"},
                )
                continue

            try:
                # 解码 Base64 音频
                audio_bytes = base64.b64decode(audio_base64)
                audio_io = BytesIO(audio_bytes)

                # 执行识别
                result = transcribe_audio(audio_io, language=language, use_itn=use_itn)

                # 发送响应
                response = {
                    "success": True,
                    "text": result["text"],
                    "language": result["language"],
                    "duration": result["duration"],
                }
                if keep_labels:
                    response["label_text"] = result["label_text"]

                await manager.send_message(websocket, response)

            except Exception as e:
                await manager.send_message(
                    websocket,
                    {"success": False, "message": f"识别失败：{str(e)}"},
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] 错误：{e}")
        manager.disconnect(websocket)


# ─── 启动入口 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
