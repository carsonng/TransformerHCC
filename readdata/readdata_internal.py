import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch.nn.functional as F


class Classification(Dataset):
    def __init__(self, mode='train', root_dir='data810_expand1'):
        super().__init__()
        self.mode = mode
        self.root_dir = root_dir

        self.patient_data = {
            'hcc': {
                'small_train': ['baozhongxic', 'cailiyuan', 'chenguanshui', 'chenjiang', 'chenjiayana',
                                'chenmingchangb',
                                'chenxikun', 'dengpeiyoub', 'fanghaizhou', 'fengzhihong', 'hanweimina', 'hanying',
                                'heruguanga', 'huangchengpeng', 'huangxianyue', 'huangyuhuab', 'jianghouming',
                                'kongzhiqin',
                                'lianganxianb', 'lianghaibo', 'liangweiqiang', 'liangxiaofeng', 'lichangqib',
                                'lihanxiong',
                                'lijianwei', 'linjinjib', 'linjinjic', 'liquanhai', 'liudayin', 'liufanlong',
                                'liuweiguang', 'liyudongb', 'lizhongmeib', 'luocunqianb', 'luoquanguoa', 'luozhizhao',
                                'luxiaoping', 'oukaihua', 'shishengmin', 'tangguohua', 'wangmingguang',
                                'wangmingguangc',
                                'wangxiaowu', 'wangxiuchun', 'weichengfang', 'wenzhennan', 'wujisheng', 'xieshenquan',
                                'xionghao', 'yangjianhongb', 'yangmulan', 'yangyongqian', 'yaojinmaob', 'yutianyong',
                                'cengguangming', 'cengweisen', 'zhengzhiyuan', 'zhoushangqing', 'zhoushidia',
                                'zouqingjiang', 'caikaizheng', 'chenlibing', 'chenyutian', 'fangdengwen', 'fushaoman',
                                'huangziying -a', 'huangziying -b'],
                'small_test': ['chendingwei', 'chenweilin', 'dengpeiyouc', 'hanweiminb', 'huangshenglong', 'hufeng',
                               'leixinnana', 'liaozhenxuan', 'likongxin', 'liqianjun', 'liuxiangwen', 'lizhongmeia',
                               'luhuanliang', 'luojian', 'raoliangping', 'tanxiuming', 'tianbing', 'wangziganga',
                               'wuxiaoming', 'yeyutai', 'zhangzhaohe', 'zhongpeixing', 'zhongshunmei', 'zhoushidib',
                               'chengweiyi', 'chenyuxianga', 'huangshima'],
                'large_train': ['anquan', 'biyan', 'caixilina', 'cenjianggu', 'chenhuiping', 'chenjiacun',
                                'chenjinshui', 'chenrichoua', 'chenweihong', 'chenweiminga', 'chenwen', 'chenyongsheng',
                                'chenyuetian', 'chenzhiwei', 'chenzhongxin', 'chenzibing', 'cuiyangxiu', 'denglianjun',
                                'dengpeiyou', 'dengshizhonga', 'dengwendao', 'dengyinyinga', 'fengjiwen', 'fengruiming',
                                'ganqimei', 'gefuliang', 'guziyang', 'hanpingan', 'heruguangb', 'heweidong',
                                'heyueqiang',
                                'huangcaidi', 'huangchengyue', 'huangjiajian', 'huangjianhua', 'huangjingliang',
                                'huangmiaochenga', 'huangshenghui', 'huangshudi', 'huangsonglaia', 'huangtianyanga',
                                'huangyuhua', 'huangzhicheng', 'huhaohua', 'jiangjianguo', 'jiangjiufen',
                                'lianganxiana',
                                'lianghaize', 'lianghe', 'liangsaozai', 'liangyihuana', 'lichangqia', 'lihaihuia',
                                'lihongxia', 'lijianwena', 'likangrena', 'limaoa', 'linbinglin', 'lingwencai',
                                'linxizhonga',
                                'liqingren', 'liruifang', 'lishixiang', 'liuheming', 'liuhongyan', 'liujiahe',
                                'liuqingtang',
                                'liuyinqiu', 'liyingguo', 'liyoumao', 'liyudonga', 'longguangxiang', 'luocunqian',
                                'luoliying', 'luoquanguoc', 'luyuzhena', 'maishengguang', 'qiaoshengjun', 'qinruiwen',
                                'tangruijian', 'tangyanzhena', 'tanzuyia', 'tanzuyib', 'wangenhuia', 'wuhairong',
                                'wuxiaopeng', 'wuxuan', 'wuyuanbiao', 'wuzhijuan', 'xiaojianrong', 'xiaozuokun',
                                'xuweisong',
                                'yangjianfenga', 'yangmulian', 'yangzhengjie', 'yanshulan', 'yezhuo',
                                'yuanzongwua', 'cengzhaoronga', 'zhangjinlun', 'zhangrenjun', 'zhangshenglu',
                                'zhaopeng',
                                'zhongweihong', 'zhongwen', 'zouriyue', 'yaojinmaoa', 'chenmingchang'],
                'large_test': ['baozhongxia', 'caiyuzhen', 'chenchengjiu', 'chenjiayanb', 'chenshulin', 'chenweimingb',
                               'chenyizhong', 'chenyuhui', 'daixiafeng', 'dengtailiang', 'dongruqiang', 'guomaozi',
                               'hexingliang', 'huangfuxiang', 'huangjixian', 'huangrongnan', 'huangshukui',
                               'huangxinzong', 'huangzhiwei', 'laishuangxionga', 'liaohuoyang', 'liguanzhao',
                               'likangrenb',
                               'linfeng', 'liqingquan', 'liqiubo', 'liubo', 'liuhuowen', 'liukangduo', 'liyazhen',
                               'liyuanfeia', 'lujinkuna', 'luweiquan', 'panxusheng', 'qiuyia', 'tanxiaoyun',
                               'wenruntang',
                               'wuxingwen', 'wuzhilong', 'yangjianhong', 'yanpeixing', 'yezhong', 'cengguorong',
                               'zhangxie',
                               'zhongdehuaa', 'hejinsheng']
            },
            'xgl': {
                'small_train': ['caijianwei', 'chenhaiquan', 'chenyongcheng', 'fuchuanzhen', 'ganhaoji', 'guanxunqiang',
                                'hanyan', 'hejinhua', 'hexingda', 'huangjianhuab', 'huangxiongjie', 'hufujian',
                                'huhenglong', 'huhuiyong', 'hujiancong', 'jinqinzhong', 'leiyi', 'liangzhiqin',
                                'lilijun',
                                'limaob', 'lishujuan', 'liugang', 'liujiandong', 'liweilong', 'lixiande', 'lixiaolai',
                                'lizhendang', 'luohanmin', 'luoquanguob', 'luyanlin', 'mojianzao', 'pengyongbiao',
                                'wangzigangb', 'wengeliang', 'wubinbiao', 'wujunqing', 'xiaoguohe', 'yangjingyao',
                                'yangsujie', 'yangzongtao', 'yuanhaizhang', 'yuyiheng', 'yuzhihao', 'zhanghui',
                                'zhengluhuan', 'zhengyaying', 'zhongshunmeib', 'zhurenteng'],
                'small_test': ['chennafeng', 'fangyanhong', 'gaozhiyong', 'helibo', 'huangmingjie', 'hujiancheng',
                               'lianglizhub', 'lihaihuib', 'liqiang', 'liuhuming', 'liwenzheng', 'longhuijian',
                               'moliewei',
                               'qishaolin', 'tangjun', 'wangyanna', 'wuxiang', 'yubiao', 'cengyunqiang', 'zhangyusheng',
                               'zhengronggui'],
                'large_train': ['caisonggui', 'caiweiting', 'caizhuofang', 'chenchengwenb', 'chenjiandong',
                                'chenjinshuib',
                                'chenqingquan', 'chenshidong', 'chenwanzhu', 'dengjieying', 'fanwenhua', 'fenghualang',
                                'guohanxiong', 'hedeqianga', 'heguiyin', 'hongyangde', 'huangmiaochengb',
                                'huangshaohui',
                                'huangsonglaib', 'huoxinlin', 'jiangjiasheng', 'jiangshifeng', 'liangjunb', 'liaowubin',
                                'lijian', 'lijuanjuan', 'linjinsheng', 'linjunyuan', 'liweijun', 'lixinbing',
                                'luolixia',
                                'luolousheng', 'mengtongtong', 'pengjiyong', 'qiuyib', 'ruanjiaolan', 'suchunlan',
                                'suyongshou', 'tangyanzhenb', 'taobinghua', 'wangenhuib', 'wangmingguangb', 'wangxia',
                                'wuchunlong', 'wuqinghui', 'wuzhangyue', 'xiaoshuang', 'xuqian', 'yezhixiong',
                                'yiyuechu',
                                'yuanzongwub', 'zhangdaoxian', 'zhanglei', 'zhangnengpiao'],
                'large_test': ['caixilinb', 'chengziyang', 'chenmugen', 'duzehui', 'guoxiangyong', 'huangkui',
                               'huangyunqiang', 'jianghai', 'liaoqingfeng', 'lijianpei', 'lixin', 'lizirong',
                               'machenlin',
                               'moxiuchao', 'shihesheng', 'tanghui', 'tianguirong', 'wangqiankun', 'wuhongguang',
                               'xianglifang', 'yangzhuhui', 'cengxiangxiong', 'zhangyiyue']
            },
            'dn': {
                'small_train': ['caowenkao', 'chenhuanhao', 'dengshizhongb', 'heyuanpeng', 'huangchengpengb',
                                'huangshenglongb', 'laishuangxiongb', 'liangheb', 'liaoweichao', 'lijianweib',
                                'likangrenc',
                                'linguanghui', 'liucaiyoub', 'lixianmei', 'liyuanfeib', 'luyuzhenb', 'xiedongxing',
                                'yanyaofeng', 'cengguangmingb', 'zhangzhijian'],
                'small_test': ['guchunfa', 'leixinnanb', 'lihaiquan', 'linxizhongb', 'lizhongmeic', 'pangtaihe',
                               'yangjianfeng', 'zhanggouwa'],
                'large_train': ['baozhongxib', 'chenbaoling', 'chenrichoub', 'chenshuilin', 'chenwei', 'chenyuxiangb',
                                'hedeqiangb', 'hongshuhua', 'huanghai', 'huangtianyangb', 'hujinwen', 'jiangqishui',
                                'lailiangyuan', 'lianglizhua', 'liangyihuanb', 'lihao', 'liquanhaib', 'liuweihuang',
                                'lujinkunb', 'ruanhuaman', 'wangdawen', 'xielianfeng', 'yangjianfengb', 'zhangwenbiao'],
                'large_test': ['chenliuguang', 'dengyinyingb', 'heshaoling', 'huangzhongwen', 'hujinwenb', 'laiyongxin',
                               'liucaiyou', 'tanglizhi', 'yanjin', 'zhongdehuab']
            },
            'fnh': {
                'small_train': ['chenshijie', 'hewenyu', 'huangwenhaob', 'maichuyan', 'songjiqiang', 'wangjun',
                                'wuyansen',
                                'yaohanyuan'],
                'small_test': ['gandongsheng', 'linxingliang', 'wangdongsheng', 'zhaoxueping'],
                'large_train': ['chenjiaying', 'chenyan', 'chenyongqiang', 'gaomanchan', 'gehaotian', 'hejiajun',
                                'huangjiandong', 'huangwenhaoa', 'huangxiaohong', 'kuangjiewen', 'liangruijing',
                                'linshaobo',
                                'linyawen', 'liufangfang', 'liuxueli', 'wangshujun', 'wangyunchuan', 'xuweituan',
                                'yaotongye', 'cengzhaorongb'],
                'large_test': ['caoying', 'chenshaozhong', 'fengjianwei', 'hufeipei', 'linshumei', 'wangquan',
                               'yanghuashou',
                               'zhongxiaoli']
            }
        }

        self.train_data = []
        self.test_data = []

        for category in self.patient_data:
            for size_type in ['small_train', 'large_train']:
                self.train_data.extend(self.patient_data[category][size_type])
            for size_type in ['small_test', 'large_test']:
                self.test_data.extend(self.patient_data[category][size_type])

        self.benign_patients = []
        self.hcc_patients = []
        for category in ['xgl', 'dn', 'fnh']:
            for size_type in ['small_train', 'large_train', 'small_test', 'large_test']:
                self.benign_patients.extend(self.patient_data[category][size_type])
        for size_type in ['small_train', 'large_train', 'small_test', 'large_test']:
            self.hcc_patients.extend(self.patient_data['hcc'][size_type])

        selected_data = self.train_data if mode == 'train' else self.test_data

        self.voxel_paths = [
            [os.path.join(root_dir, name, str(i), 'tumor.nii.gz') for i in range(1, 6)]
            for name in selected_data
        ]

        print(f'total have {len(self.train_data + self.test_data)} labels')

    def _load_nii_file(self, nii_path):
        image = sitk.ReadImage(nii_path)
        return sitk.GetArrayFromImage(image)

    def _resize_3d(self, image, size=(14, 14, 14)):
        image = image.astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(image_tensor, size=size, mode='trilinear', align_corners=False)
        return resized.squeeze(0).squeeze(0).cpu().numpy()

    def _normalize(self, image):
        mean, std = np.mean(image), np.std(image)
        return (image - mean) / std

    def _random_flip(self, image, axis, p=0.5):
        if np.random.random() <= p:
            if axis == 0:
                return image[::-1, ...]
            elif axis == 1:
                return image[:, ::-1, ...]
            elif axis == 2:
                return image[..., ::-1]
        return image

    def _apply_transforms(self, voxel_stack):
        for i in range(len(voxel_stack)):
            for axis in [0, 1, 2]:
                voxel_stack[i] = self._random_flip(voxel_stack[i], axis, p=0.5)
        return voxel_stack

    def __getitem__(self, item):
        voxels = []
        for path in self.voxel_paths[item]:
            voxel = self._load_nii_file(path)
            voxel = self._normalize(voxel)
            voxel = self._resize_3d(voxel)
            voxel = np.expand_dims(voxel, axis=0)
            voxels.append(voxel)

        voxel_stack = np.concatenate(voxels, axis=0)

        if self.mode == 'train':
            voxel_stack = self._apply_transforms(voxel_stack)

        patient_name = self.voxel_paths[item][0].split(os.sep)[-3]
        label = 0 if patient_name in self.benign_patients else 1

        voxel1, voxel2, voxel3, voxel4, voxel5 = np.split(voxel_stack, 5, axis=0)

        if self.mode == 'test':
            return voxel1, voxel2, voxel3, voxel4, voxel5, label, patient_name
        else:
            return voxel1, voxel2, voxel3, voxel4, voxel5, label

    def __len__(self):
        return len(self.voxel_paths)









