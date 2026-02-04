# Copyright (c) Opendatalab. All rights reserved.
"""
Convert DeepSeek-OCR-2 output to MinerU middle_json format.
"""

import re
import hashlib
from typing import List, Dict, Any
from io import BytesIO

from loguru import logger
from mineru.data.data_reader_writer import DataWriter


def _generate_image_hash(image) -> str:
    """Generate hash for image filename."""
    buf = BytesIO()
    image.save(buf, format='PNG')
    return hashlib.md5(buf.getvalue()).hexdigest()[:16]


def _parse_markdown_to_blocks(markdown: str, page_idx: int) -> List[Dict[str, Any]]:
    """
    Parse markdown text into structured blocks.

    Args:
        markdown: Markdown text from DeepSeek
        page_idx: Page index

    Returns:
        List of block dictionaries
    """
    blocks = []
    lines = markdown.split('\n')

    current_text = []
    block_idx = 0

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            if current_text:
                blocks.append({
                    'type': 'text',
                    'text': '\n'.join(current_text),
                    'page_idx': page_idx,
                    'block_idx': block_idx,
                })
                current_text = []
                block_idx += 1
            continue

        # Check for headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if header_match:
            if current_text:
                blocks.append({
                    'type': 'text',
                    'text': '\n'.join(current_text),
                    'page_idx': page_idx,
                    'block_idx': block_idx,
                })
                current_text = []
                block_idx += 1

            level = len(header_match.group(1))
            blocks.append({
                'type': 'text',
                'text': header_match.group(2),
                'text_level': level,
                'page_idx': page_idx,
                'block_idx': block_idx,
            })
            block_idx += 1
            continue

        # Check for images
        img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
        if img_match:
            if current_text:
                blocks.append({
                    'type': 'text',
                    'text': '\n'.join(current_text),
                    'page_idx': page_idx,
                    'block_idx': block_idx,
                })
                current_text = []
                block_idx += 1

            blocks.append({
                'type': 'image',
                'alt': img_match.group(1),
                'src': img_match.group(2),
                'page_idx': page_idx,
                'block_idx': block_idx,
            })
            block_idx += 1
            continue

        # Check for tables (simple detection)
        if stripped.startswith('|') and stripped.endswith('|'):
            # Collect table rows
            table_lines = [stripped]
            # This is a simplified table detection
            blocks.append({
                'type': 'table',
                'text': stripped,
                'page_idx': page_idx,
                'block_idx': block_idx,
            })
            block_idx += 1
            continue

        # Check for formulas
        if stripped.startswith('$') or stripped.startswith('\\['):
            if current_text:
                blocks.append({
                    'type': 'text',
                    'text': '\n'.join(current_text),
                    'page_idx': page_idx,
                    'block_idx': block_idx,
                })
                current_text = []
                block_idx += 1

            blocks.append({
                'type': 'equation',
                'text': stripped,
                'page_idx': page_idx,
                'block_idx': block_idx,
            })
            block_idx += 1
            continue

        # Regular text
        current_text.append(stripped)

    # Don't forget remaining text
    if current_text:
        blocks.append({
            'type': 'text',
            'text': '\n'.join(current_text),
            'page_idx': page_idx,
            'block_idx': block_idx,
        })

    return blocks


def deepseek_result_to_middle_json(
    results: List[Dict[str, Any]],
    image_writer: DataWriter,
) -> Dict[str, Any]:
    """
    Convert DeepSeek OCR results to MinerU middle_json format.

    Args:
        results: List of page results from DeepSeek
        image_writer: Writer for saving images

    Returns:
        Dictionary in middle_json format
    """
    middle_json = {
        'pdf_info': [],
        '_parse_type': 'deepseek',
    }

    all_markdown = []

    for page_result in results:
        page_idx = page_result['page_idx']
        markdown = page_result['markdown']
        image = page_result.get('image')

        # Parse markdown into blocks
        blocks = _parse_markdown_to_blocks(markdown, page_idx)

        # Save page image if available
        page_image_path = None
        if image is not None:
            img_hash = _generate_image_hash(image)
            img_filename = f"page_{page_idx}_{img_hash}.png"
            buf = BytesIO()
            image.save(buf, format='PNG')
            image_writer.write(img_filename, buf.getvalue())
            page_image_path = f"images/{img_filename}"

        page_info = {
            'page_idx': page_idx,
            'blocks': blocks,
            'markdown': markdown,
            'page_image': page_image_path,
            'width': image.width if image else 0,
            'height': image.height if image else 0,
        }

        middle_json['pdf_info'].append(page_info)
        all_markdown.append(markdown)

    # Store combined markdown
    middle_json['full_markdown'] = '\n\n---\n\n'.join(all_markdown)

    return middle_json
