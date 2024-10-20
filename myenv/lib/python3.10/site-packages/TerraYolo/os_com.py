import sys
if 'google.colab' in sys.modules:
    from .colab_os_com import colab_os_com as os_com
else:
    from .non_colab_os_com import non_colab_os_com as os_com