import os
import logging

if os.environ.get('ICECUBE_DATA'):
    data_dir = os.environ.get('ICECUBE_DATA')
else:
    data_dir = os.path.join(os.environ.get('HOME'), 'ICECUBE_DATA', 'icecube_10year_ps')
years = ('40', '59', '79', '86_I', '86_II', '86_III', '86_IV', '86_V', '86_VI', '86_VII')
pdf_store_dir = os.path.join(os.environ.get('HOME'), 'ICECUBE_DATA', 'pdf_store')

logger = logging.getLogger(__name__)
logger.info('IceCube data dir: %s. PDF store: %s', data_dir, pdf_store_dir)