import os
import time

class Logger:
    def __init__(self,config):
        self.config=config
        self.log_file=os.path.join(config.logs_dir,"train.log")
        self.start_time=time.time()
        self.buffer=[]
    def log(self,msg,print_msg=True):
        elapsed=time.time()-self.start_time
        line=f"[{elapsed:7.2f}s] {msg}"
        if print_msg:
            print(line)
        self.buffer.append(line)
    def save_log(self):
        with open(self.log_file,"a",encoding="utf-8") as f:
            for line in self.buffer:
                f.write(line+"\n")
        self.buffer=[]