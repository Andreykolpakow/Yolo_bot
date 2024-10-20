from subprocess import Popen, PIPE, STDOUT
from IPython.core.magic import register_line_magic


@register_line_magic
def colab_os_com(command):

    os.environ["PYTHONUNBUFFERED"] = "1"

    with Popen(
            command, stdout=PIPE, shell=True, stderr=STDOUT, bufsize=1, close_fds=True
    ) as process:
        for line in iter(process.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))


# old
'''
#import types
def colab_os_com(obj, command, log_f=True):
    if log_f == True:
        os.system(command + '>>' + obj.log_f_name)
    else:
        !{command}
    return
setattr(self, 'os_com', types.MethodType(colab_os_com, TerraYoloV5))
'''
'''
import shlex
from subprocess import STDOUT, check_call

        else:
            # !{command}
            #current_working_dir = os.getcwd()
            #print('current_working_dir', current_working_dir)
            #tmp_str = input()
            #process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
            #process = subprocess.Popen(command,cwd=current_working_dir, stdout = PIPE, stderr = PIPE, shell = True)#, encoding='utf8'
            #process = subprocess.Popen(command, stdout = PIPE, stderr = PIPE, shell = False)#, encoding='utf8'
            #process = subprocess.Popen(shlex.split(command), cwd=current_working_dir, stdout = subprocess.PIPE, encoding='utf8')#
            #process = subprocess.Popen(shlex.split(command), shell = False, stdout = subprocess.PIPE, encoding='utf8')
            #process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)


            while True:
                output = process.stdout.readline()
                print(output)
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode("utf-8"))
            rc = process.poll()
'''