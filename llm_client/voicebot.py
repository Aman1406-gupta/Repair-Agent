import asyncio
import json
import base64
import struct
import ssl
import time
import websockets
import openai

MODEL      = "gpt-realtime"
WS_URL     = f"wss://api.openai.com/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24_000                           # mono 16-bit PCM, 24 kHz

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode   = ssl.CERT_NONE


class RealtimeVoiceBot:
    """Lightweight wrapper around OpenAI's realtime websocket API."""

    # ---------- static audio helpers ---------------------------------------
    @staticmethod
    def float32_to_pcm16(samples):
        return b"".join(
            struct.pack("<h", max(-32767, min(32767, int(x * 32767))))
            for x in samples
        )

    @staticmethod
    def pcm16_to_b64(pcm):
        return base64.b64encode(pcm).decode("ascii")

    @staticmethod
    def b64_to_pcm16(b64):
        return base64.b64decode(b64)

    # ----------------------------------------------------------------------
    def __init__(self,
                 instructions: str = None,
                 voice: str = "sage",
                 api_key: str | None = None,
                 *,
                 turn_detection: None | dict = None,
                 extra_session_fields: dict | None = None):
        """
        • instructions : system prompt
        • voice        : TTS voice name ('alloy', 'sage', …)
        • turn_detection : None for push-to-talk, or {"type":"vad"} etc.
        • extra_session_fields : any other keys you want inside session.update
        """
        self.api_key = "enter own api key"

        self.voice   = voice
        self.instr   = instructions
        self.turn_detection = turn_detection
        self.extra   = extra_session_fields or {}

        self._ws = None
        self.last_ttft   = None   # latency of most recent response
        self.last_total  = None
        self.tts_client = openai.OpenAI(api_key=self.api_key)

    def tts_audio(
        self,
        text: str,
        *,
        model: str  = "gpt-4o-mini-tts",
        voice: str  = "alloy",
        response_format: str = "pcm",
        autoplay: bool = True):

        pcm_buffer = bytearray()
        with self.tts_client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format,
        ) as response:
            # iter_bytes() yields raw audio bytes chunk-by-chunk
            for chunk in response.iter_bytes():
                pcm_buffer.extend(chunk)

        return bytes(pcm_buffer)

    # ----------------------------------------------------------------------
    async def connect(self):
        if self._ws:
            return                               # already open
        self._ws = await websockets.connect(
            WS_URL, additional_headers=[("Authorization",
                                         f"Bearer {self.api_key}")],
            ssl=ssl_ctx
        )
        await self._ws.recv()                    # initial handshake
        await self._update_session(first=True)

    async def _update_session(self, first=False):
        session_body = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input":  {"turn_detection": self.turn_detection},
                    "output": {"voice": self.voice, "speed": 1.0}
                },
                **self.extra
            }
        }
        if self.instr:
            session_body['session']['instructions'] = self.instr


        await self._ws.send(json.dumps(session_body))
        upd_message = await self._ws.recv()
        return upd_message

    async def update_session(self, *, instructions=None, voice=None, **extra):
        if instructions: self.instr = instructions
        if voice:        self.voice = voice
        self.extra.update(extra)
        await self._update_session()

    # ----------------------------------------------------------------------
    def send_text(self, text: str):
        asyncio.run(self._asend_text(text))

    async def _asend_text(self, text: str):
        await self._connect_if_needed()
        await self._ws.send(json.dumps({
            "type":"conversation.item.create",
            "item":{
                "type":"message",
                "role":"user",
                "content":[{"type":"input_text", "text":text}]
        }}))


    def send_audio(self, pcm16_bytes: bytes):
        asyncio.run(self._asend_audio(pcm16_bytes))

    async def _asend_audio(self, pcm16_bytes: bytes):
        await self._connect_if_needed()
        await self._ws.send(json.dumps({
            "type":"conversation.item.create",
            "item":{
                "type":"message",
                "role":"user",
                "content":[{"type":"input_audio",
                            "audio": self.pcm16_to_b64(pcm16_bytes)}]
        }}))


    def send_function_output(self, call_id,output):
        asyncio.run(self._asend_function_output(call_id,output))

    async def _asend_function_output(self, call_id,output):
        await self._connect_if_needed()
        await self._ws.send(json.dumps({
            "type":"conversation.item.create",
            "item":{
                "type":"function_call_output",
                "call_id":call_id,
                "output":output
        }}))


    # ----------------------------------------------------------------------
    def receive_response(self,instructions=None,tools=None,output_modalities=("audio",)):
        return asyncio.run(self._areceive_response(instructions,tools,output_modalities))

    async def _areceive_response(self,instructions=None,tools=None,output_modalities=("audio",)):
        request_body = {
            "type":"response.create",
            "response":{"output_modalities": list(output_modalities),
                        "tool_choice":"auto"}
        }
        if instructions:
            request_body["response"]["instructions"] = instructions
        if tools:
            request_body["response"]["tools"] = tools
        await self._ws.send(json.dumps(request_body))
        text, transcript, audio,function_calls,usage = "", "", bytearray(),[],{}
        start = time.perf_counter()
        ttft = None
        while True:
            evt = json.loads(await self._ws.recv())
            etype = evt["type"]
            now   = time.perf_counter()
            if etype == "response.output_audio.delta" and ttft is None:
                ttft = now - start
            if   etype == "response.output_text.delta":
                text += evt["delta"]
            elif etype == "response.output_audio_transcript.delta":
                transcript += evt["delta"]
            elif etype == "response.output_audio.delta":
                audio += self.b64_to_pcm16(evt["delta"])
            elif etype == "response.done":
                total = now - start
                for output in evt['response']['output']:
                    if output['type']=='function_call':
                        function_calls.append({"name":output['name'],"call_id":output['call_id'],"arguments":output['arguments']})
                    usage = evt['response']['usage']
                break

        self.last_ttft, self.last_total = ttft, total
        return text, transcript, bytes(audio),function_calls,usage

    # ----------------------------------------------------------------------
    async def _connect_if_needed(self):
        if not self._ws:
            await self.connect()

    async def close(self):
        if self._ws :
            await self._ws.close()
