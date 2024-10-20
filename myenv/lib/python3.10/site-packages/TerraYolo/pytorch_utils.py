import os
import torch


def torch_load(terra_model,
               repo_or_dir='ultralytics/yolov5',
               sub_model='yolov5s',
               source='github',
               weight_path='',
               pretrained=True, force_reload=False, verbose=True):
    """

    :param repo_or_dir:
    :param sub_model:
    :param source:
    :param weight_path:
    :param pretrained:
    :param force_reload:
    :param verbose:
    :return:
    Note: If ``source`` is 'github', ``repo_or_dir`` is expected to be
    of the form ``repo_owner/repo_name[:ref]`` with an optional
    ref (a tag or a branch).

    If ``source`` is 'local', ``repo_or_dir`` is expected to be a
    path to a local directory. - do not implemented
    """

    wd = os.getcwd()
    os.chdir(terra_model.work_dir)
    if sub_model == 'custom':
        if os.path.isfile(weight_path):
            terra_model.hub_model = torch.hub.load(
                repo_or_dir, sub_model,
                path=weight_path,
                force_reload=force_reload,
                verbose=verbose
            )
            return
        else:
            print(f'веса не {weight_path} найдены, пробуем загрузить')

    terra_model.hub_model = torch.hub.load(
        repo_or_dir, sub_model, source=source,
        pretrained=pretrained,
        force_reload=force_reload,
        verbose=verbose
    )
    os.chdir(wd)
    return
