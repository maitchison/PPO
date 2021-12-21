"""
Very simple file based (unsafe) mutex lock with short timeout.
"""

import os
import time
import stat
from pathlib import Path
import unicodedata
import re

LOCK_PATH = os.path.expanduser("~/.cache/")
LOCK_AGE_LIMIT = 120 # in seconds


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

class Mutex:

    def __init__(self, key, timeout=60.0):
        """
        only enabled if mutex is not None
        """
        self.enabled = key is not None
        self.key = slugify(key) if key is not None else ""
        self.timeout = timeout
        self.wait_time = 0.0

    def __enter__(self):
        if self.enabled:
            self.wait_time = aquire_lock(self.key, self.timeout)
        return self

    def __exit__(self, type, value, traceback):
        if self.enabled:
            release_lock(self.key)


def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]


def aquire_lock(key, timeout=60.0):
    """
    Get lock, timeout is in seconds.
    """

    start_time = time.time()
    lock_file = Path(os.path.join(LOCK_PATH, key+".lock"))

    if lock_file.exists():
        # if the lock is old just ignore it.
        if file_age_in_seconds(lock_file.absolute()) > LOCK_AGE_LIMIT:
            lock_file.touch()
            return 0.0

        # otherwise, wait until the lock is available
        while lock_file.exists() and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        wait_time = time.time() - start_time

        if lock_file.exists():
            return timeout
        else:
            lock_file.touch()
            return wait_time

    lock_file.touch()
    return 0.0


def release_lock(key):
    try:
        os.remove(os.path.join(LOCK_PATH, key + ".lock"))
    except:
        pass


