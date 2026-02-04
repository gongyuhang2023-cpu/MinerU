# Copyright (c) Opendatalab. All rights reserved.
# DeepSeek-OCR-2 Backend Implementation
"""
DeepSeek-OCR-2 integration for MinerU.
Model: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
"""

import os
import time
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger

from mineru.utils.pdf_image_tools import load_images_from_pdf
from mineru.data.data_reader_writer import DataWriter
from .model_output_to_middle_json import deepseek_result_to_middle_json


class DeepSeekModelSingleton:
    """Singleton for DeepSeek-OCR-2 model to avoid repeated loading."""
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Load DeepSeek-OCR-2 model.

        Args:
            model_path: Path to local model or HuggingFace model ID
            device: Device to run on (cuda, cpu)

        Returns:
            tuple: (model, tokenizer)
        """
        if self._model is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers and torch:\n"
                "pip install torch transformers"
            )

        if model_path is None:
            model_path = "deepseek-ai/DeepSeek-OCR-2"

        logger.info(f"[DeepSeek] Loading model from {model_path}...")
        start_time = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Try to use flash attention if available
        try:
            self._model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"[DeepSeek] Flash attention not available: {e}")
            logger.info("[DeepSeek] Falling back to default attention...")
            self._model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=True
            )

        import torch
        self._model = self._model.eval().to(device).to(torch.bfloat16)

        elapsed = time.time() - start_time
        logger.info(f"[DeepSeek] Model loaded in {elapsed:.2f}s")

        return self._model, self._tokenizer


def doc_analyze(
    pdf_bytes: bytes,
    image_writer: DataWriter,
    model_path: Optional[str] = None,
    device: str = "cuda",
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
    **kwargs,
) -> dict:
    """
    Analyze PDF document using DeepSeek-OCR-2.

    Args:
        pdf_bytes: PDF file content as bytes
        image_writer: Writer for saving extracted images
        model_path: Path to DeepSeek model
        device: Device to run on
        base_size: Base size for image processing
        image_size: Image size for OCR
        crop_mode: Whether to use crop mode
        **kwargs: Additional arguments

    Returns:
        dict: Analysis result with markdown content per page
    """
    import torch

    # Load model
    singleton = DeepSeekModelSingleton()
    model, tokenizer = singleton.get_model(model_path, device)

    # Convert PDF to images
    logger.info("[DeepSeek] Converting PDF to images...")
    images = load_images_from_pdf(pdf_bytes)
    logger.info(f"[DeepSeek] Got {len(images)} pages")

    # Process each page
    results = []
    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    for page_idx, img in enumerate(images):
        logger.info(f"[DeepSeek] Processing page {page_idx + 1}/{len(images)}...")

        # Save image to temp file (DeepSeek requires file path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name, format='PNG')
            tmp_path = tmp.name

        try:
            # Run inference
            with torch.no_grad():
                result = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=False,
                )

            # Extract markdown from result
            if isinstance(result, dict):
                markdown = result.get('markdown', result.get('text', str(result)))
            elif isinstance(result, str):
                markdown = result
            else:
                markdown = str(result)

            results.append({
                'page_idx': page_idx,
                'markdown': markdown,
                'image': img,
            })

        except Exception as e:
            logger.error(f"[DeepSeek] Error processing page {page_idx + 1}: {e}")
            results.append({
                'page_idx': page_idx,
                'markdown': f"[Error processing page: {e}]",
                'image': img,
            })

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    # Save images and convert to middle_json format
    return deepseek_result_to_middle_json(results, image_writer)


async def aio_doc_analyze(
    pdf_bytes: bytes,
    image_writer: DataWriter,
    model_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> dict:
    """
    Async version of doc_analyze.
    Note: DeepSeek model inference is synchronous,
    so this wraps the sync version.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: doc_analyze(
            pdf_bytes, image_writer, model_path, device, **kwargs
        )
    )
