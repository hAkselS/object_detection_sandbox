import logging 
import time 

logger = logging.getLogger()

logging.basicConfig(filename='test_code/loggile.log', encoding='utf-8', level=logging.DEBUG)

logger.info('heres another statement for log file 1')

logger.error('and more from log file 1')

x = 0 
while x < 10:
    logger.debug('prog 1')
    time.sleep(1)