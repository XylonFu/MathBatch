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
        response = item.get(self.response_field, "")
        if isinstance(response, str):
            item[self.response_field] = response.strip()
        return item

    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List]:
        responses = []
        for item in batch:
            processed = self.process_item(item)
            responses.append(processed.get(self.response_field))
        return {"responses": responses}


class CleanExtractTagPostProcessor(StripPostProcessor):
    def __init__(self, response_field: str):
        super().__init__(response_field=response_field)
        self.response_field = response_field

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        response = item.get(self.response_field, "")
        if isinstance(response, str):
            item[self.response_field] = response.replace('Extracted Answer: ', '').strip()
        return item
