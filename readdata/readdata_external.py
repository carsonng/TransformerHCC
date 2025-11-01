import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch.nn.functional as F


class Classification(Dataset):
    def __init__(self, root_dir='./external_data810_expand1'):
        super().__init__()
        self.root_dir = root_dir

        self.patient_data = {
            'hcc': ['1', '2', '15', '25', '27', '38', '66', '78', '78-2', '80', '83', '100', '106', '127', '127-2',
                    '132',
                    '140', '163', '163-2', '182', '184', '186', '187', '189', '195', '202', '223', '223-2', '247',
                    '255',
                    '257', '259', '263', '270', '287', '292', '341', 'L27', 'L35', 'L85', 'L102', 'L127', 'L137',
                    'L168',
                    'L187', 'L196', 'L203', 'L220', 'L224', 'L235', 'L253', 'L253-2', 'L253-3', 'L257', 'L267', 'L302',
                    'L315', 'L324', 'L346', 'L348', 'L348-2', 'L364', 'L385', 'L433', 'L443', 'L443-2', 'L453', 'L473',
                    'L473-2', 'L486', 'L496', 'L537', 'L542', 'L552', 'L567', 'L579', 'L580', 'L583', 'L584', 'L586',
                    'huguixi', 'lianzhanglai', 'linbaoren', 'liujingfan', 'liyanhao', 'luoguangshan', 'luohongdi',
                    'qianxiuzhen', 'renxuedian', 'tianguixiang', 'wangwenming', 'wangzhendan', 'weibinxian -a',
                    'weibinxian -b', 'wentong', 'wukefang', 'wuxin', 'xieshaoqing', 'zhangdechao', 'zhangguoqiang',
                    'zhanzhenxiong', 'zhaoaifen', 'zhengjinyang', 'zhengwang', 'zhoucanyan', 'zhouguoliang',
                    'zhuoxiaowen', 'zoubaiqi'],
            'xgl': ['23', '89', '92', '114', '114-2', '139', '247-2', '290', '296', '299', '299-2', '321',
                    'L25', 'L37', 'L42', 'L58', 'L74-2', 'L86', 'L114', 'L114-2', 'L118', 'L118-2', 'L123', 'L138',
                    'L146', 'L147', 'L154', 'L174', 'L185', 'L239', 'L248', 'L257-2', 'L258', 'L285', 'L293',
                    'L315-2', 'L331-2', 'L351', 'L372', 'L374', 'L378', 'L444', 'L501', 'L507', 'L544', 'L546',
                    'L546-2', 'L547', 'L563'],
            'dn': ['2-2', '83-2', '184-2', '302', '315', '315-2', '324', 'L27-2', 'L41', 'L131', 'L203-2',
                   'L261', 'L377', 'L435', 'L435-2', 'L453-2', 'L552-2', 'L552-3', 'L581', 'L585','59', '183', '272', '272-2', 'L302-2', 'L302-3'],

            'fnh': ['20', '39', '68', '68-2', '152', '159', '177', 'L26', 'L50', 'L67', 'L142', 'L207',
                    'L218', 'L221', 'L221-2', 'L243', 'L317', 'L331', 'L412', 'L412-2', 'L426', 'L426-2',
                    'L431', 'L484', 'L485', 'L498', 'L545']
        }

        self.benign_patients = self.patient_data['xgl'] + self.patient_data['dn'] + self.patient_data['fnh']
        self.hcc_patients = self.patient_data['hcc']

        self.all_patients = self.benign_patients + self.hcc_patients
        self.voxel_paths = [
            [os.path.join(root_dir, name, str(i), 'tumor.nii.gz') for i in range(1, 6)]
            for name in self.all_patients
        ]

        print(f'total have {len(self.all_patients)} patients')

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

    def __getitem__(self, item):
        voxels = []
        for path in self.voxel_paths[item]:
            voxel = self._load_nii_file(path)
            voxel = self._normalize(voxel)
            voxel = self._resize_3d(voxel)
            voxel = np.expand_dims(voxel, axis=0)
            voxels.append(voxel)

        voxel_stack = np.concatenate(voxels, axis=0)

        patient_name = self.voxel_paths[item][0].split(os.sep)[-3]
        label = 0 if patient_name in self.benign_patients else 1

        voxel1, voxel2, voxel3, voxel4, voxel5 = np.split(voxel_stack, 5, axis=0)

        return voxel1, voxel2, voxel3, voxel4, voxel5, label, patient_name

    def __len__(self):
        return len(self.all_patients)



