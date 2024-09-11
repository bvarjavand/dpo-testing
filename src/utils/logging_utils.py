import logging
import sys
import datasets
import transformers
import determined as det

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        format=det.LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    transformers.utils.logging.set_verbosity_info()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    return logger