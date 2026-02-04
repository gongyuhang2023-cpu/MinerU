# Copyright (c) Opendatalab. All rights reserved.
"""
Generate markdown output from DeepSeek middle_json.
"""

from typing import Dict, Any, List

from mineru.utils.enum_class import MakeMode


def union_make(
    middle_json: Dict[str, Any],
    make_mode: MakeMode = MakeMode.MM_MD,
    img_parent_path: str = "",
) -> str:
    """
    Generate markdown from DeepSeek middle_json.

    Since DeepSeek directly outputs markdown, this function
    mainly handles image path adjustments.

    Args:
        middle_json: The middle_json from DeepSeek
        make_mode: Output mode (MM_MD for markdown)
        img_parent_path: Parent path for images

    Returns:
        Markdown string
    """
    # DeepSeek already provides markdown, so we can use it directly
    if 'full_markdown' in middle_json:
        return middle_json['full_markdown']

    # Fallback: combine page markdowns
    pages = middle_json.get('pdf_info', [])
    markdown_parts = []

    for page in pages:
        page_md = page.get('markdown', '')
        if page_md:
            markdown_parts.append(page_md)

    return '\n\n---\n\n'.join(markdown_parts)


def make_content_list(middle_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate content list from middle_json.

    Args:
        middle_json: The middle_json from DeepSeek

    Returns:
        List of content items
    """
    content_list = []

    for page in middle_json.get('pdf_info', []):
        for block in page.get('blocks', []):
            item = {
                'type': block.get('type', 'text'),
                'text': block.get('text', ''),
                'page_idx': block.get('page_idx', 0),
            }

            if 'text_level' in block:
                item['text_level'] = block['text_level']

            if 'bbox' in block:
                item['bbox'] = block['bbox']

            content_list.append(item)

    return content_list
