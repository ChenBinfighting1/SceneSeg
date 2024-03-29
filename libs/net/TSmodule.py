# encoding: utf-8
from torch import nn
from libs.net import generateNet as net_gener


class TSmodule(nn.Module):
    """
    TSmodule
    创建TS的模型类
    """

    def __init__(self, args):
        super(TSmodule, self).__init__()
        self.Teacher_arg = args.T
        self.Student_arg = args.S
        self.Teacher = net_gener.__dict__[self.Teacher_arg.type](self.Teacher_arg)
        self.Student = net_gener.__dict__[self.Student_arg.type](self.Student_arg)
        
        self.mode = args.mode
        
        self.featuremap = args.featuremap

    def forward(self, x, x_T = None):
        if x_T is None:
            x_T = x
        if self.mode == "reconstruction" and self.featuremap == "aspp":
            t_aspp = self.Teacher.forward_till_aspp(x_T)
            s_aspp = self.Student.forward_till_aspp(x)
            s_result = self.Student.catFeat_to_predict(
                self.Student.aspp_to_catFeat(s_aspp))
            return (t_aspp, s_aspp, s_result)
        elif self.mode == "reconstruction" and self.featuremap == "concat":
            t_aspp = self.Teacher.forward_till_aspp(x_T)
            s_aspp = self.Student.forward_till_aspp(x)
            t_concat = self.Teacher.aspp_to_catFeat(t_aspp)
            s_concat = self.Student.aspp_to_catFeat(s_aspp)
            s_result = self.Student.catFeat_to_predict(s_concat)
            return (t_concat, s_concat, s_result)
        elif self.mode == "reconstruction&KLDiv" and self.featuremap == "concat":
            t_aspp = self.Teacher.forward_till_aspp(x_T)
            s_aspp = self.Student.forward_till_aspp(x)
            # print('t_aspp',t_aspp.size())
            # print("s_aspp",s_aspp.size())
            t_concat = self.Teacher.aspp_to_catFeat(t_aspp)
            s_concat, s_concat_2 = self.Student.aspp_to_catFeat(s_aspp)
            # print('t_concat', t_concat.size())
            # print("s_concat", s_concat.size())
            t_result = self.Teacher.catFeat_to_predict(t_concat)
            s_result, s_result_2 = self.Student.catFeat_to_predict(s_concat)
            # print('t_result', t_result.size())
            # print("s_result", s_result.size())
            return (t_concat, s_concat_2, s_result, t_result, s_result_2)
        elif self.mode == "reconstruction" and self.featuremap == "concat&aspp":
            t_aspp = self.Teacher.forward_till_aspp(x_T)
            s_aspp = self.Student.forward_till_aspp(x)
            t_concat = self.Teacher.aspp_to_catFeat(t_aspp)
            s_concat = self.Student.aspp_to_catFeat(s_aspp)
            s_result = self.Student.catFeat_to_predict(s_concat)
            return (t_aspp, s_aspp, t_concat, s_concat, s_result)
        elif self.mode == "KLDiv":
            t_result = self.Student.catFeat_to_predict(self.Teacher.aspp_to_catFeat(self.Teacher.forward_till_aspp(x_T)))
            s_result = self.Student.catFeat_to_predict(self.Student.aspp_to_catFeat(self.Student.forward_till_aspp(x)))
            return (t_result, s_result)