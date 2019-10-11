import torch.nn as nn
class LRsc_poly:
    def __init__(self, power, max_itr, lr, optimizer):
        self.power = power
        self.max_itr = max_itr
        self.lr = lr
        self.optimizer = optimizer

    def __call__(self, itr):
        now_lr = self.lr * (1 - itr / (self.max_itr + 1)) ** self.power
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]['lr'] = now_lr
        else:
            self.optimizer.param_groups[0]['lr'] = now_lr
            self.optimizer.param_groups[1]['lr'] = 10 * now_lr
        return now_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class Fix:
    def __init__(self, power, max_itr, lr, optimizer):
        self.power = power
        self.max_itr = max_itr
        self.lr = lr
        self.optimizer = optimizer

    def __call__(self, itr):
        return self.lr


class Param_change:
    def __init__(self, lr, net=None, distill=None):

        if distill is not None and distill.S_head == "direct":
            self.params = [
                    {'params': self.get_params(net.module.Student, key='1x'), 'lr': lr},
                    {'params': self.get_params(net.module.Student, key='10x'), 'lr': lr*10},
                    {'params': self.get_params(net.module.Teacher, key='1x'), 'lr': 0.},
                    {'params': self.get_params(net.module.Teacher, key='10x'), 'lr': 0.}
                ]
            net.module.Teacher.require_grad = False
        elif distill is not None and distill.S_head == "teacher":
            self.params = [
                    {'params': self.get_params(net.module.Student, key='studentbb'), 'lr': lr},
                    {'params': self.get_params(net.module.Student, key='studentaspp'), 'lr': lr*10},
                    {'params': self.get_params(net.module.Student, key='teacherHead'), 'lr': 0},
                    {'params': self.get_params(net.module.Teacher, key='1x'), 'lr': 0.},
                    {'params': self.get_params(net.module.Teacher, key='10x'), 'lr': 0.}
                ]
            net.module.Teacher.require_grad = False
        elif distill is not None and distill.S_head == "teacherconcat":
            self.params = [
                    {'params': self.get_params(net.module.Student, key='studentbb'), 'lr': lr},
                    {'params': self.get_params(net.module.Student, key='studentasppconcat'), 'lr': lr*10},
                    {'params': self.get_params(net.module.Student, key='teacherHeadconcat'), 'lr': 0},
                    {'params': self.get_params(net.module.Teacher, key='1x'), 'lr': 0.},
                    {'params': self.get_params(net.module.Teacher, key='10x'), 'lr': 0.}
                ]
            net.module.Teacher.require_grad = False
        else:
            self.params = [
                    {'params': self.get_params(net.module, key='1x'), 'lr': lr},
                    {'params': self.get_params(net.module, key='10x'), 'lr': lr*10}
                ]


    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == 'studentaspp':
                if 'aspp' in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == 'studentasppconcat':
                if ('aspp' in m[0] or 'shortcut_conv' in m[0]) and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == 'studentbb':
                if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == 'teacherHead':
                if 'backbone' not in m[0] and 'aspp' not in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == 'teacherHeadconcat':
                if 'backbone' not in m[0] and 'aspp' not in m[0] and 'shortcut_conv' not in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p

class Param_default:
    def __init__(self, lr, net=None):
        self.params = [{'params':net.parameters(), 'lr':lr}]
