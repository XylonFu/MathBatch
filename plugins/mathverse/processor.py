import re
from typing import Dict, Any

from cores.processor import TextOnlyDataPreProcessor, StripPostProcessor


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


class CleanExtractTagPostProcessor(StripPostProcessor):
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


class ScoreAnswerPreProcessor(TextOnlyDataPreProcessor):
    def __init__(self, query_field: str, question_field: str, answer_field: str, demo_prompt: str):
        super().__init__(query_field=query_field)
        self.question_field = question_field
        self.answer_field = answer_field
        self.demo_prompt = demo_prompt.strip()

    def _format_query(self, question, answer, extraction):
        full_prompt = self.demo_prompt.format(question=question, gt=answer, extraction=extraction)
        return full_prompt

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        formated_query = self._format_query(question=item[self.question_field],
                                            answer=item[self.answer_field],
                                            extraction=item[self.query_field])
        return {"prompt": formated_query}


class CleanJudgeTagPostProcessor(StripPostProcessor):
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        response = item[self.response_field].replace("Judgement:", "").strip()

        think_pattern = r'<think>.*?</think>(.*)'
        match = re.search(think_pattern, response, re.DOTALL)

        if match:
            content_after_think = match.group(1).strip()
            item[self.response_field] = int(content_after_think)
        else:
            item[self.response_field] = int(response)

        return item
