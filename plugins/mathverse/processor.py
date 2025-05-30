import re
from typing import Dict, Any, List

from cores.base import BaseDataProcessor
from cores.processor import TextOnlyDataPreProcessor


class ExtractionAnswerPreProcessor(TextOnlyDataPreProcessor):
    def __init__(self, query_field: str, demo_prompt: str):
        super().__init__(query_field=query_field)
        self.demo_prompt = demo_prompt.strip()

    def _format_query(self, response):
        test_prompt = f"Model response: '{response}'\nExtracted Answer: "
        return f"{self.demo_prompt}\n\n{test_prompt}"

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        formated_query = self._format_query(item[self.query_field])
        return {"prompt": formated_query}


class StripPostProcessor(BaseDataProcessor):
    def __init__(self, response_field: str):
        super().__init__(field_name=response_field)
        self.response_field = response_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item[self.response_field] = item[self.response_field].strip()
        return item

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        responses = [self.process_item(item)[self.response_field] for item in batch]
        return {"responses": responses}


class CleanExtractTagPostProcessor(StripPostProcessor):
    def __init__(self, response_field: str):
        super().__init__(response_field=response_field)
        self.response_field = response_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        response = item[self.response_field].replace('Extracted Answer: ', '').strip()

        think_pattern = r'<think>.*?</think>(.*)'
        match = re.search(think_pattern, response, re.DOTALL)

        if match:
            content_after_think = match.group(1).strip()
            item[self.response_field] = content_after_think
        else:
            item[self.response_field] = response

        return item
