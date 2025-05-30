# cores/executor.py
import logging
import os

from .base import (
    BaseDataLoader,
    BaseDataFilter,
    BaseDataProcessor,
    BaseDataBatcher,
    BaseDataSaver
)

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Executes the processing pipeline"""

    def __init__(
            self,
            data_loader: BaseDataLoader,
            data_processor: BaseDataProcessor,
            data_filter: BaseDataFilter,
            batch_processor: BaseDataBatcher,
            data_saver: BaseDataSaver
    ):
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.data_filter = data_filter
        self.batch_processor = batch_processor
        self.data_saver = data_saver

    def execute_pipeline(
            self,
            input_file: str,
            output_file: str,
            rerun: bool
    ) -> None:
        """Executes the processing pipeline"""
        # Load input data
        input_data = self.data_loader.load_dataset(input_file)
        if not input_data:
            logger.error(f"No data loaded from: {input_file}")
            return

        logger.info(f"Loaded {len(input_data)} items from: {input_file}")

        # Initialize output data
        output_data = self.data_loader.initialize_output_dataset(
            input_data, output_file, rerun)

        # Filter items needing processing
        pending_items = self.data_filter.filter_unprocessed_items(output_data)
        pending_count = len(pending_items)

        if not pending_count:
            logger.info("No pending items to process")
            if not os.path.exists(output_file):
                self.data_saver.save_data(output_data)
            return

        logger.info(f"Processing {pending_count} pending items")

        # Process data in batches
        processed_count = 0
        for batch in self.batch_processor.process_batches(
                pending_items,
                self.data_processor,
                self.data_loader.response_field
        ):
            processed_count += len(batch)
            logger.info(f"Processed: {processed_count}/{pending_count} items")

            # Update and save checkpoint
            self.data_saver.save_checkpoint(output_data)

        # Final save
        if self.data_saver.save_data(output_data):
            logger.info(f"Results saved to: {output_file}")
            # Clean up temporary file
            if os.path.exists(f"{output_file}.tmp"):
                os.remove(f"{output_file}.tmp")
