import logging 
import time

logger = logging.getLogger()

logging.basicConfig(filename='test_code/loggile.log', encoding='utf-8', level=logging.DEBUG)

logger.info('added a statement to my log 2')

logger.error('and another statement 2')

x = 0 
while x < 5:
    logger.debug('prog 2')
    time.sleep(3)
    x = x +1 