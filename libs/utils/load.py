# encoding: utf-8
import torch

def loadParam(net, cfg, distill):
    if hasattr(distill, "TeacherModel"):
        net_dict = net.Teacher.state_dict()
        checkpoint = torch.load(distill.TeacherModel, map_location="cpu")
        pretrained_dict = {}
        for k, v in checkpoint.items():
            k = k.replace("module.", "")
            k = k.replace("Teacher.", "")
            if(k in net_dict) and (v.shape == net_dict[k].shape):
                pretrained_dict[k] = v
            else:
                print(k,"do not load param")
        net_dict.update(pretrained_dict)
        net.Teacher.load_state_dict(net_dict)
        if distill.S_head == "teacher":
            net_dict = net.Student.state_dict()
            checkpoint = torch.load(distill.TeacherModel)
            pretrained_dict = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                k = k.replace("Teacher.", "")
                if(k in net_dict) and (v.shape == net_dict[k].shape) and 'backbone' not in k and 'aspp' not in k:
                    pretrained_dict[k] = v
                else:
                    print(k,"do not load param")
            net_dict.update(pretrained_dict)
            net.Student.load_state_dict(net_dict)
        elif distill.S_head == "teacherconcat":
            net_dict = net.Student.state_dict()
            checkpoint = torch.load(distill.TeacherModel)
            pretrained_dict = {}
            for k, v in checkpoint.items():
                k = k.replace("module.", "")
                k = k.replace("Teacher.", "")
                if(k in net_dict) and (v.shape == net_dict[k].shape) and 'backbone' not in k and 'aspp' not in k and 'shortcut_conv' not in k:
                    pretrained_dict[k] = v
                else:
                    print(k,"do not load param")
            net_dict.update(pretrained_dict)
            net.Student.load_state_dict(net_dict)
    else:
        # 实际不能使用
        if cfg.TRAIN_CKPT:
            net_dict = net.state_dict()
            pretrained_dict = torch.load(cfg.TRAIN_CKPT)
            pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if(k in net_dict) and (v.shape == net_dict[k].shape):
                    pretrained_dict[k] = v
                else:
                    print(k,"do not load param")
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)