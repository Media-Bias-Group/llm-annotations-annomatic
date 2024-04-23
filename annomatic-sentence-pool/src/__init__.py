import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="\x1b[32;1m" + "%(message)s (%(filename)s:%(lineno)d)" + "\x1b[0m",
)