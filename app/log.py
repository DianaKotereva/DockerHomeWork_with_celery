import logging
import sys

handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    format='%(name)s - %(message)s',
    level=logging.INFO,
    handlers=handlers
)

log = logging.getLogger(__name__)
