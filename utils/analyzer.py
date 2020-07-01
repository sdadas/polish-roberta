import atexit
import os
import signal
from typing import Tuple, List

from pexpect import popen_spawn
from pexpect.popen_spawn import PopenSpawn
import urllib.request

class PolishAnalyzer(object):

    def __init__(self):
        self.analyzer: PopenSpawn = self.__run_analyzer()
        self.analyzer.delimiter = "\n"
        atexit.register(lambda: self.analyzer.kill(signal.SIGTERM))

    def __run_analyzer(self) -> PopenSpawn:
        dir = os.path.dirname(os.path.realpath(__file__))
        jar = os.path.join(dir, "polish-simple-analyzer.jar").replace('\\', '/')
        self.__download_analyzer(jar)
        process: PopenSpawn = popen_spawn.PopenSpawn('java -jar %s' % (jar,), encoding='utf-8')
        return process

    def __download_analyzer(self, jar: str):
        if os.path.exists(jar): return
        release = "https://github.com/sdadas/polish-simple-analyzer/releases/download/v0.1/polish-simple-analyzer.jar"
        urllib.request.urlretrieve(release, jar)

    def analyze(self, sentence: str) -> Tuple[List, List]:
        escaped = sentence.encode('unicode-escape').decode('ascii').replace('\\x', '\\u00')
        self.analyzer.sendline(escaped)
        tokens = self.analyzer.readline().strip().encode().decode('unicode-escape')
        lemmas = self.analyzer.readline().strip().encode().decode('unicode-escape')
        return tokens.strip().split(), lemmas.strip().split()
