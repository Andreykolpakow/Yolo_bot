from subprocess import Popen, PIPE, STDOUT
from IPython.core.magic import register_line_magic

def non_colab_os_com(command):
    with Popen(
        command, stdout=PIPE, shell=True, stderr=STDOUT, bufsize=1, close_fds=True
    ) as process:
        for line in iter(process.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))

'''
import types
@register_line_magic
def colab_os_com(obj, command, log_f=True):
    if log_f == True:
        os.system(command + '>>' + obj.log_f_name)
    else:
        colab_os_com(command)
    return
setattr(my_terra_yolov5, 'os_com', types.MethodType(colab_os_com, TerraYoloV5))
'''