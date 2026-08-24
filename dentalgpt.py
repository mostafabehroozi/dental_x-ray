"""Small Hugging Face runner for the original DentalGPT checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class DentalGPTRunner:
    def __init__(
        self,
        model_id: str = "Eric3200/DentalGPT-7B-1026",
        processor_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        hf_token: str | None = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            processor_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            token=hf_token,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            attn_implementation="eager",
            token=hf_token,
        ).eval()

    @torch.inference_mode()
    def ask(self, image_path: str, question: str) -> dict:
        image_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        started = time.perf_counter()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        latency = time.perf_counter() - started

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return {"raw_answer": answer, "latency_seconds": round(latency, 3)}
